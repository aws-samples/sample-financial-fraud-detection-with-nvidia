"""Neptune graph client replacing CSV-based graph I/O for fraud detection."""

import json
import os
import time

import boto3
import numpy as np
import pandas as pd
import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest


class NeptuneClient:
    """Client for Amazon Neptune, replacing CSV-based graph storage."""

    def __init__(self, endpoint=None, port=8182, region=None, iam_auth=True):
        self.endpoint = endpoint or os.environ.get("NEPTUNE_ENDPOINT")
        self.port = int(port or os.environ.get("NEPTUNE_PORT", 8182))
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.iam_auth = iam_auth
        self.base_url = f"https://{self.endpoint}:{self.port}"
        if iam_auth:
            session = boto3.Session(region_name=self.region)
            self.credentials = session.get_credentials().get_frozen_credentials()

    def _signed_request(self, method, path, data=None, retries=5):
        url = f"{self.base_url}{path}"
        for attempt in range(retries):
            headers = {"Content-Type": "application/json"}
            body = json.dumps(data) if data else None
            if self.iam_auth:
                req = AWSRequest(method=method, url=url, data=body, headers=headers)
                SigV4Auth(self.credentials, "neptune-db", self.region).add_auth(req)
                headers.update(dict(req.headers))
            try:
                resp = requests.request(
                    method, url, json=data, headers=headers, timeout=600
                )
                if resp.status_code >= 500:
                    # Neptune returns 5xx when overwhelmed or scaling up
                    print(f"  Neptune returned {resp.status_code}: {resp.text[:500]}")
                    if attempt == retries - 1:
                        resp.raise_for_status()
                    wait = 2 ** (attempt + 1)
                    print(f"  Retrying in {wait}s (attempt {attempt + 1}/{retries})...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except (
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
            ):
                if attempt == retries - 1:
                    raise
                wait = 2 ** (attempt + 1)
                print(
                    f"  Neptune request failed, retrying in {wait}s (attempt {attempt + 1}/{retries})..."
                )
                time.sleep(wait)

    def execute_opencypher(self, query, parameters=None):
        data = {"query": query}
        if parameters:
            data["parameters"] = json.dumps(parameters)
        return self._signed_request("POST", "/opencypher", data)

    def clear_graph(self):
        self.execute_opencypher("MATCH (n) DETACH DELETE n")

    # --- Batch writes ---

    def write_users_batch(self, user_df, batch_size=50):
        rows = [
            {"uid": int(idx), "features": json.dumps(row.values.tolist())}
            for idx, row in user_df.iterrows()
        ]
        for i in range(0, len(rows), batch_size):
            self._batch_create_nodes("User", "uid", rows[i : i + batch_size])

    def write_merchants_batch(self, merchant_df, batch_size=50):
        rows = [
            {"mid": int(idx), "features": json.dumps(row.values.tolist())}
            for idx, row in merchant_df.iterrows()
        ]
        for i in range(0, len(rows), batch_size):
            self._batch_create_nodes("Merchant", "mid", rows[i : i + batch_size])

    def write_transactions_batch(
        self, edge_index, edge_attr_df, edge_label_df, batch_size=100
    ):
        rows = [
            {
                "uid": int(edge_index[0, i]),
                "mid": int(edge_index[1, i]),
                "features": json.dumps(edge_attr_df.iloc[i].values.tolist()),
                "label": int(edge_label_df.iloc[i].values[0]),
                "idx": i,
            }
            for i in range(edge_index.shape[1])
        ]
        for i in range(0, len(rows), batch_size):
            self._batch_create_edges(rows[i : i + batch_size])

    def _batch_create_nodes(self, label, id_key, rows):
        self.execute_opencypher(
            f"UNWIND $rows AS row CREATE (n:{label} {{{id_key}: row.{id_key}, features: row.features}})",
            {"rows": rows},
        )

    def _batch_create_edges(self, rows):
        self.execute_opencypher(
            "UNWIND $rows AS row "
            "MATCH (u:User {uid: row.uid}), (m:Merchant {mid: row.mid}) "
            "CREATE (u)-[:TRANSACTION {features: row.features, fraud: row.label, tx_idx: row.idx}]->(m)",
            {"rows": rows},
        )

    # --- Reads ---

    def read_users(self):
        result = self.execute_opencypher(
            "MATCH (u:User) RETURN u.uid AS uid, u.features AS features ORDER BY u.uid"
        )
        records = result.get("results", [])
        if not records:
            return pd.DataFrame()
        return pd.DataFrame([json.loads(r["features"]) for r in records]).astype(
            np.float32
        )

    def read_merchants(self):
        result = self.execute_opencypher(
            "MATCH (m:Merchant) RETURN m.mid AS mid, m.features AS features ORDER BY m.mid"
        )
        records = result.get("results", [])
        if not records:
            return pd.DataFrame()
        return pd.DataFrame([json.loads(r["features"]) for r in records]).astype(
            np.float32
        )

    def read_transactions(self):
        result = self.execute_opencypher(
            "MATCH (u:User)-[t:TRANSACTION]->(m:Merchant) "
            "RETURN u.uid AS uid, m.mid AS mid, t.features AS features, "
            "t.fraud AS fraud, t.tx_idx AS tx_idx ORDER BY t.tx_idx"
        )
        records = result.get("results", [])
        if not records:
            return (
                np.empty((2, 0), dtype=np.int64),
                pd.DataFrame(),
                pd.DataFrame({"Fraud": []}),
            )
        uids = [r["uid"] for r in records]
        mids = [r["mid"] for r in records]
        edge_index = np.array([uids, mids], dtype=np.int64)
        edge_attr = pd.DataFrame([json.loads(r["features"]) for r in records]).astype(
            np.float32
        )
        edge_labels = pd.DataFrame({"Fraud": [r["fraud"] for r in records]})
        return edge_index, edge_attr, edge_labels

    def load_hetero_graph(self):
        """Load full graph from Neptune, matching the dict format of CSV-based load_hetero_graph()."""
        user_df = self.read_users()
        merchant_df = self.read_merchants()
        edge_index, edge_attr, edge_labels = self.read_transactions()

        out = {}
        if not user_df.empty:
            out["x_user"] = user_df.to_numpy(dtype=np.float32)
            out["feature_mask_user"] = np.zeros(user_df.shape[1], dtype=np.int32)
        if not merchant_df.empty:
            out["x_merchant"] = merchant_df.to_numpy(dtype=np.float32)
            out["feature_mask_merchant"] = np.zeros(
                merchant_df.shape[1], dtype=np.int32
            )
        if not edge_attr.empty:
            out["edge_index_user_to_merchant"] = edge_index
            out["edge_attr_user_to_merchant"] = edge_attr.to_numpy(dtype=np.float32)
            out["edge_feature_mask_user_to_merchant"] = np.zeros(
                edge_attr.shape[1], dtype=np.int32
            )
        out["edge_label_user_to_merchant"] = edge_labels
        return out

    def get_graph_stats(self):
        def _count(q):
            return self.execute_opencypher(q).get("results", [{}])[0].get("cnt", 0)

        return {
            "users": _count("MATCH (u:User) RETURN count(u) AS cnt"),
            "merchants": _count("MATCH (m:Merchant) RETURN count(m) AS cnt"),
            "transactions": _count(
                "MATCH ()-[t:TRANSACTION]->() RETURN count(t) AS cnt"
            ),
            "fraud": _count(
                "MATCH ()-[t:TRANSACTION]->() WHERE t.fraud = 1 RETURN count(t) AS cnt"
            ),
        }

#!/usr/bin/env python3
"""
Fraud Detection Endpoint Test Script

This script tests the deployed SageMaker endpoint running NVIDIA's GNN-based
fraud detection model on Triton Inference Server.

The Model:
    The model is a Graph Neural Network (GNN) that predicts whether a financial
    transaction is fraudulent. It operates on a bipartite graph where:
    - Users (card holders) are one set of nodes
    - Merchants are another set of nodes
    - Transactions are edges connecting users to merchants

    For each transaction (edge), the model outputs a fraud probability between 0 and 1.

Input Features:
    - x_user (num_users, 13): User/card features (binary-encoded card ID)
    - x_merchant (num_merchants, 24): Merchant features (encoded merchant name + MCC code)
    - edge_index_user_to_merchant (2, num_transactions): Graph connectivity [user_ids, merchant_ids]
    - edge_attr_user_to_merchant (num_transactions, 38): Transaction features including:
        * Amount (scaled)
        * Error codes (one-hot encoded)
        * Chip usage (binary)
        * Location info (City, Zip - binary encoded)
        * MCC code (binary encoded)

    - feature_mask_*: Masks for Shapley value computation (group correlated features)
    - COMPUTE_SHAP: Boolean flag to enable/disable Shapley explanation

Output:
    - PREDICTION (num_transactions, 1): Fraud probability for each transaction
        * Close to 0.0 = likely legitimate
        * Close to 1.0 = likely fraudulent
        * Typical threshold: 0.5

    When COMPUTE_SHAP=True, also returns:
    - shap_values_user: Feature importance for user attributes
    - shap_values_merchant: Feature importance for merchant attributes
    - shap_values_user_to_merchant: Feature importance for transaction attributes

Usage:
    python test_endpoint.py
    python test_endpoint.py --endpoint-name fraud-detection-endpoint-v2
    python test_endpoint.py --profile zjacobso+nvidia-Admin --benchmark
"""

import argparse
import json
import sys
import time

import boto3
import numpy as np


# Feature dimensions from the TabFormer dataset preprocessing
USER_FEATURE_DIM = 13       # Binary-encoded card ID
MERCHANT_FEATURE_DIM = 24   # Merchant name + MCC (binary encoded)
EDGE_FEATURE_DIM = 38       # Transaction attributes (Amount, Error, Chip, City, Zip, MCC)


def make_test_data(num_merchants=5, num_users=7, num_transactions=3):
    """
    Create synthetic test data representing a small transaction graph.

    This simulates a scenario with:
    - num_users card holders making purchases
    - num_merchants receiving payments
    - num_transactions individual purchases to evaluate

    In production, these features would come from the preprocessing pipeline
    that transforms raw transaction data (amounts, locations, timestamps, etc.)
    into the encoded feature vectors the model expects.
    """
    return {
        # Node features
        "x_user": np.random.randn(num_users, USER_FEATURE_DIM).astype(np.float32),
        "x_merchant": np.random.randn(num_merchants, MERCHANT_FEATURE_DIM).astype(np.float32),

        # Graph structure: which user transacted with which merchant
        # Row 0 = user indices, Row 1 = merchant indices
        "edge_index_user_to_merchant": np.array([
            np.random.randint(0, num_users, num_transactions),
            np.random.randint(0, num_merchants, num_transactions)
        ], dtype=np.int64),

        # Transaction features (amount, location, payment method, etc.)
        "edge_attr_user_to_merchant": np.random.randn(num_transactions, EDGE_FEATURE_DIM).astype(np.float32),

        # Shapley computation settings
        "COMPUTE_SHAP": np.array([False], dtype=np.bool_),
        "feature_mask_user": np.zeros(USER_FEATURE_DIM, dtype=np.int32),
        "feature_mask_merchant": np.zeros(MERCHANT_FEATURE_DIM, dtype=np.int32),
        "edge_feature_mask_user_to_merchant": np.zeros(EDGE_FEATURE_DIM, dtype=np.int32),
    }


def numpy_to_triton_inputs(data):
    """Convert numpy arrays to Triton inference request format."""
    dtype_map = {
        np.float32: "FP32",
        np.float64: "FP64",
        np.int32: "INT32",
        np.int64: "INT64",
        np.bool_: "BOOL",
    }
    return [
        {
            "name": name,
            "shape": list(arr.shape),
            "datatype": dtype_map.get(arr.dtype.type, "FP32"),
            "data": arr.flatten().tolist(),
        }
        for name, arr in data.items()
    ]


def invoke_endpoint(runtime, endpoint_name, data, compute_shap=False):
    """Send inference request to SageMaker endpoint."""
    data = data.copy()
    data["COMPUTE_SHAP"] = np.array([compute_shap], dtype=np.bool_)

    outputs = [{"name": "PREDICTION"}]
    if compute_shap:
        outputs.extend([
            {"name": "shap_values_merchant"},
            {"name": "shap_values_user"},
            {"name": "shap_values_user_to_merchant"},
        ])

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps({"inputs": numpy_to_triton_inputs(data), "outputs": outputs}),
    )
    return json.loads(response["Body"].read().decode())


def parse_response(response):
    """Parse Triton response into dict of numpy arrays."""
    return {
        output["name"]: np.array(output["data"]).reshape(output["shape"])
        for output in response.get("outputs", [])
    }


def format_prediction(prob, threshold=0.5):
    """Format a single prediction as human-readable text."""
    label = "FRAUD" if prob > threshold else "LEGIT"
    confidence = prob if prob > threshold else (1 - prob)
    return f"{label} ({confidence*100:.1f}% confidence)"


def print_predictions(predictions, edge_index, threshold=0.5):
    """Print predictions in a human-readable format."""
    print("\n  Transaction Results:")
    print("  " + "-" * 50)

    num_fraud = 0
    for i, prob in enumerate(predictions.flatten()):
        user_id = edge_index[0, i]
        merchant_id = edge_index[1, i]
        result = format_prediction(prob, threshold)
        is_fraud = prob > threshold
        num_fraud += is_fraud

        indicator = "X" if is_fraud else " "
        print(f"  [{indicator}] Tx {i+1}: User {user_id} -> Merchant {merchant_id}")
        print(f"       Prediction: {result} (raw: {prob:.4f})")

    print("  " + "-" * 50)
    print(f"  Summary: {num_fraud}/{len(predictions)} flagged as potential fraud")


def test_health(sm_client, endpoint_name):
    """Check endpoint health."""
    print(f"\n{'='*60}")
    print("Test 1: Endpoint Health Check")
    print('='*60)

    try:
        info = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = info["EndpointStatus"]
        print(f"  Endpoint: {endpoint_name}")
        print(f"  Status: {status}")

        if status == "InService":
            print("  Result: PASSED")
            return True
        else:
            print(f"  Result: FAILED - Status is {status}, expected InService")
            return False
    except Exception as e:
        print(f"  Result: FAILED - {e}")
        return False


def test_inference(runtime, endpoint_name):
    """Test basic inference with human-readable output."""
    print(f"\n{'='*60}")
    print("Test 2: Basic Inference")
    print('='*60)

    try:
        data = make_test_data(num_merchants=3, num_users=4, num_transactions=5)

        print("\n  Graph Structure:")
        print(f"    Users (card holders): {data['x_user'].shape[0]}")
        print(f"    Merchants: {data['x_merchant'].shape[0]}")
        print(f"    Transactions to evaluate: {data['edge_attr_user_to_merchant'].shape[0]}")

        response = invoke_endpoint(runtime, endpoint_name, data, compute_shap=False)
        result = parse_response(response)

        predictions = result.get("PREDICTION")
        if predictions is not None:
            print_predictions(predictions, data["edge_index_user_to_merchant"])
            print("\n  Result: PASSED")
            return True
        else:
            print("  Result: FAILED - No PREDICTION in response")
            return False
    except Exception as e:
        print(f"  Result: FAILED - {e}")
        return False


def test_shapley(runtime, endpoint_name):
    """Test inference with Shapley value explanations."""
    print(f"\n{'='*60}")
    print("Test 3: Explainability (Shapley Values)")
    print('='*60)

    print("\n  Shapley values explain which features contributed to the")
    print("  fraud prediction. Higher absolute values = more influence.")

    try:
        data = make_test_data(num_merchants=2, num_users=3, num_transactions=1)
        response = invoke_endpoint(runtime, endpoint_name, data, compute_shap=True)
        result = parse_response(response)

        print("\n  Explanation Outputs:")
        for name, arr in result.items():
            if name == "PREDICTION":
                prob = arr.flatten()[0]
                print(f"    {name}: {format_prediction(prob)} (raw: {prob:.4f})")
            elif name.startswith("shap_values_"):
                feature_type = name.replace("shap_values_", "")
                # Shapley values are aggregated per feature group
                total_contribution = np.sum(arr)
                print(f"    {name}: {arr.shape} -> total contribution: {total_contribution:.4f}")

        print("\n  Note: Shapley computation is expensive. In production,")
        print("  only request explanations for flagged transactions.")
        print("\n  Result: PASSED")
        return True
    except Exception as e:
        print(f"  Result: FAILED - {e}")
        return False


def test_benchmark(runtime, endpoint_name, num_requests=10):
    """Benchmark inference latency."""
    print(f"\n{'='*60}")
    print(f"Test 4: Latency Benchmark ({num_requests} requests)")
    print('='*60)

    try:
        data = make_test_data()
        latencies = []

        print("\n  Running inference requests...")
        for i in range(num_requests):
            start = time.time()
            invoke_endpoint(runtime, endpoint_name, data, compute_shap=False)
            latencies.append((time.time() - start) * 1000)
            print(f"    Request {i+1}/{num_requests}: {latencies[-1]:.0f} ms")

        print(f"\n  Latency Statistics:")
        print(f"    Mean:  {np.mean(latencies):>6.0f} ms")
        print(f"    Std:   {np.std(latencies):>6.0f} ms")
        print(f"    Min:   {np.min(latencies):>6.0f} ms")
        print(f"    Max:   {np.max(latencies):>6.0f} ms")
        print(f"    P50:   {np.percentile(latencies, 50):>6.0f} ms")
        print(f"    P95:   {np.percentile(latencies, 95):>6.0f} ms")
        print("\n  Result: PASSED")
        return True
    except Exception as e:
        print(f"  Result: FAILED - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test fraud detection endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run basic tests
  %(prog)s --benchmark                        # Include latency benchmark
  %(prog)s --endpoint-name my-endpoint-v2     # Test specific endpoint
  %(prog)s --profile my-aws-profile           # Use specific AWS profile
"""
    )
    parser.add_argument("--endpoint-name", default="fraud-detection-endpoint",
                        help="SageMaker endpoint name")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--profile", default=None, help="AWS profile name")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run latency benchmark")
    parser.add_argument("--benchmark-requests", type=int, default=10,
                        help="Number of benchmark requests")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Fraud decision threshold (default: 0.5)")
    args = parser.parse_args()

    # Setup AWS clients
    session_kwargs = {"region_name": args.region}
    if args.profile:
        session_kwargs["profile_name"] = args.profile

    session = boto3.Session(**session_kwargs)
    sm_client = session.client("sagemaker")
    runtime = session.client("sagemaker-runtime")

    print("\n" + "="*60)
    print("Fraud Detection Endpoint Test Suite")
    print("="*60)
    print(f"\nEndpoint: {args.endpoint_name}")
    print(f"Region:   {args.region}")
    print(f"Profile:  {args.profile or 'default'}")
    print(f"Threshold: {args.threshold}")
    print("\nThis model detects fraudulent financial transactions using a")
    print("Graph Neural Network that analyzes user-merchant relationships.")

    # Run tests
    results = []
    results.append(("Health Check", test_health(sm_client, args.endpoint_name)))

    if results[-1][1]:  # Only continue if health check passed
        results.append(("Basic Inference", test_inference(runtime, args.endpoint_name)))
        results.append(("Shapley Values", test_shapley(runtime, args.endpoint_name)))

        if args.benchmark:
            results.append(("Latency Benchmark",
                          test_benchmark(runtime, args.endpoint_name, args.benchmark_requests)))

    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print("="*60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"  {name}: {status}")

    print(f"\n  Total: {passed}/{total} tests passed")
    print("="*60 + "\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

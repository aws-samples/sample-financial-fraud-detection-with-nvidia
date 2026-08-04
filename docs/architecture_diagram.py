"""
Architecture diagram: GNN Fraud Detection on AWS
Generates dark and light PNG variants using the diagrams library.

Run from the project root:
    python docs/architecture_diagram.py
"""

from diagrams import Cluster, Diagram, Edge
from diagrams.aws.compute import ECR
from diagrams.aws.database import Neptune
from diagrams.aws.devtools import CloudDevelopmentKit, Codebuild
from diagrams.aws.general import Client, User
from diagrams.aws.ml import Sagemaker, SagemakerModel, SagemakerTrainingJob
from diagrams.aws.network import VPC
from diagrams.aws.security import IAM, SecretsManager
from diagrams.aws.storage import S3
from diagrams.onprem.vcs import Github


# ---------------------------------------------------------------------------
# Theming helpers
# ---------------------------------------------------------------------------

def graph_attr(dark: bool) -> dict:
    return {
        "bgcolor": "transparent",
        "dpi": "300",
        "pad": "0.8",
        "nodesep": "1.0",
        "ranksep": "1.5",
        "splines": "ortho",
        "fontname": "Helvetica Neue",
        "fontsize": "14",
        "fontcolor": "#FFFFFF" if dark else "#161D26",
    }


def edge_attr(dark: bool) -> dict:
    return {
        "color": "#FFFFFFbb" if dark else "#232F3Ebb",
        "penwidth": "1.8",
        "arrowsize": "0.7",
        "fontname": "Helvetica Neue",
        "fontsize": "9",
        "fontcolor": "#FFFFFF" if dark else "#161D26",
    }


NODE_ATTR = {"fontname": "Helvetica Neue", "fontsize": "11"}


def cluster_style(dark: bool, color: str = "#FF9900") -> dict:
    return {
        "bgcolor": "#232F3E33" if dark else "#F3F3F722",
        "pencolor": color,
        "penwidth": "2",
        "style": "rounded",
        "fontname": "Helvetica Neue Bold",
        "fontsize": "13",
        "fontcolor": "#FFFFFF" if dark else "#161D26",
        "margin": "20",
    }


# ---------------------------------------------------------------------------
# Generate diagrams
# ---------------------------------------------------------------------------

for dark in [True, False]:
    suffix = "dark" if dark else "light"
    with Diagram(
        "",
        filename=f"docs/gnn_fraud_detection_architecture_{suffix}",
        show=False,
        direction="LR",
        graph_attr=graph_attr(dark),
        node_attr=NODE_ATTR,
        edge_attr=edge_attr(dark),
    ):
        # ── Actors ────────────────────────────────────────────────────────
        developer = User("Developer")

        with Cluster("Point of Sale", graph_attr=cluster_style(dark, "#DD344C")):
            pos = Client("POS Terminal")

        # ── Source Control ────────────────────────────────────────────────
        with Cluster("Source Control", graph_attr=cluster_style(dark, "#6B7280")):
            github = Github("GitHub")

        # ── AWS Cloud ─────────────────────────────────────────────────────
        with Cluster("AWS Cloud", graph_attr=cluster_style(dark, "#FF9900")):

            # Foundation layer
            with Cluster("Foundation", graph_attr=cluster_style(dark, "#3B82F6")):
                cdk = CloudDevelopmentKit("CDK")
                iam = IAM("IAM Roles")
                s3 = S3("S3 Buckets")
                secrets = SecretsManager("Secrets\nManager")

            # Container image build
            with Cluster("Container Image Build", graph_attr=cluster_style(dark, "#42B4FF")):
                codebuild = Codebuild("CodeBuild")
                ecr = ECR("ECR")

            # SageMaker Studio
            with Cluster("SageMaker Studio", graph_attr=cluster_style(dark, "#A855F7")):
                studio = Sagemaker("Studio\nDomain")

            # ML Pipeline
            with Cluster("SageMaker Pipeline", graph_attr=cluster_style(dark, "#FF9900")):
                preprocess = Sagemaker("Preprocessing\nRAPIDS / cuDF")
                train = SagemakerTrainingJob("Training\nGNN + XGBoost")
                registry = SagemakerModel("Model\nRegistry")

            # Graph Database
            with Cluster("Graph Database (VPC)", graph_attr=cluster_style(dark, "#10B981")):
                neptune = Neptune("Neptune\nServerless")
                vpc = VPC("Private VPC")

            # Real-time inference
            with Cluster("Real-Time Inference", graph_attr=cluster_style(dark, "#00CC88")):
                endpoint = Sagemaker("Triton Endpoint\nGPU Auto-scaling")

        # ── Developer flows ───────────────────────────────────────────────
        developer >> Edge(label="deploy") >> cdk
        developer >> Edge(label="notebooks") >> studio

        # CDK provisions everything
        cdk >> Edge(label="provision", style="dashed") >> iam
        cdk >> Edge(label="provision", style="dashed") >> s3
        cdk >> Edge(label="provision", style="dashed") >> ecr
        cdk >> Edge(label="provision", style="dashed") >> neptune

        # ── Image build flows ─────────────────────────────────────────────
        github >> Edge(label="source") >> codebuild
        secrets >> Edge(style="dashed", label="NGC key") >> codebuild
        codebuild >> Edge(label="push images") >> ecr

        # ECR feeds container images into pipeline + inference
        ecr >> Edge(style="dashed", label="RAPIDS") >> preprocess
        ecr >> Edge(style="dashed", label="training") >> train
        ecr >> Edge(style="dashed", label="Triton") >> endpoint

        # ── ML pipeline flow ──────────────────────────────────────────────
        s3 >> Edge(label="raw data") >> preprocess
        preprocess >> Edge(label="graph\ndata") >> neptune
        preprocess >> Edge(label="features") >> s3
        neptune >> Edge(label="graph\nstructure") >> train
        s3 >> Edge(label="processed\ndata") >> train
        train >> Edge(label="model.tar.gz") >> s3
        train >> registry

        # ── Inference flow ────────────────────────────────────────────────
        s3 >> Edge(label="model\nartifacts") >> endpoint
        registry >> Edge(label="approved\nmodel") >> endpoint
        neptune >> Edge(style="dashed", label="graph\nlookup") >> endpoint

        # ── POS interaction ───────────────────────────────────────────────
        pos >> Edge(label="transaction") >> endpoint
        endpoint >> Edge(label="fraud score\n+ Shapley values") >> pos

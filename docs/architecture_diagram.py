"""
Architecture diagram: GNN Fraud Detection on SageMaker
Generates dark and light PNG variants from the mermaid specification.
"""

from diagrams import Cluster, Diagram, Edge
from diagrams.aws.compute import ECR
from diagrams.aws.devtools import CloudDevelopmentKit, Codebuild
from diagrams.aws.general import Client, User
from diagrams.aws.ml import Sagemaker, SagemakerModel, SagemakerTrainingJob
from diagrams.aws.security import IAM, SecretsManager
from diagrams.aws.storage import S3
from diagrams.onprem.vcs import Github


def graph_attr(dark):
    return {
        "bgcolor": "transparent",
        "dpi": "300",
        "pad": "0.5",
        "nodesep": "0.8",
        "ranksep": "1.2",
        "fontname": "Helvetica Neue",
        "fontsize": "14",
        "fontcolor": "#FFFFFF" if dark else "#161D26",
    }


def edge_attr(dark):
    return {
        "color": "#FFFFFFbb" if dark else "#232F3Ebb",
        "penwidth": "2.0",
        "arrowsize": "0.8",
        "fontname": "Helvetica Neue",
        "fontsize": "10",
        "fontcolor": "#FFFFFF" if dark else "#161D26",
    }


NODE_ATTR = {"fontname": "Helvetica Neue", "fontsize": "12"}


def cluster_attr(dark, border_color="#FF9900"):
    return {
        "bgcolor": "#232F3E22" if dark else "#F3F3F711",
        "pencolor": border_color,
        "penwidth": "2",
        "style": "rounded",
        "fontname": "Helvetica Neue",
        "fontsize": "13",
        "fontcolor": "#FFFFFF" if dark else "#161D26",
    }


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
        # Entry points
        developer = User("developer")
        studio = Sagemaker("studio")
        client = Client("applications")

        # Infrastructure
        cdk = CloudDevelopmentKit("cdk")
        iam = IAM("roles")
        s3 = S3("data & models")
        ecr = ECR("container registry")

        # Image Build cluster
        with Cluster("Image Build", graph_attr=cluster_attr(dark, "#42B4FF")):
            codebuild = Codebuild("codebuild")
            secrets = SecretsManager("ngc api key")
            github = Github("source")

        # SageMaker Pipeline cluster
        with Cluster("SageMaker Pipeline", graph_attr=cluster_attr(dark, "#FF9900")):
            preprocess = Sagemaker("preprocessing\nRAPIDS / cuDF")
            train = SagemakerTrainingJob("training\nGNN + XGBoost")
            registry = SagemakerModel("model registry")

        # Inference cluster
        with Cluster("Inference", graph_attr=cluster_attr(dark, "#00CC88")):
            endpoint = Sagemaker("endpoint\nTriton / GPU")

        # Developer flows
        developer >> cdk
        developer >> studio

        # CDK provisions infrastructure
        cdk >> Edge(label="provision") >> s3
        cdk >> ecr
        cdk >> iam
        cdk >> codebuild

        # Image Build flows
        github >> Edge(style="dashed", label="source") >> codebuild
        secrets >> Edge(style="dashed", label="credentials") >> codebuild
        codebuild >> Edge(label="push") >> ecr

        # ECR provides images to pipeline and inference
        ecr >> Edge(style="dashed", label="RAPIDS image") >> preprocess
        ecr >> Edge(style="dashed", label="training image") >> train
        ecr >> Edge(style="dashed", label="Triton image") >> endpoint

        # SageMaker Pipeline flow
        s3 >> Edge(label="raw data") >> preprocess
        preprocess >> Edge(label="graph data") >> s3
        s3 >> Edge(label="processed data") >> train
        train >> Edge(label="model.tar.gz") >> s3
        preprocess >> train >> registry

        # Inference flow
        s3 >> Edge(label="model artifacts") >> endpoint
        registry >> Edge(label="approved model") >> endpoint
        endpoint >> Edge(label="fraud probability\n+ Shapley values") >> client

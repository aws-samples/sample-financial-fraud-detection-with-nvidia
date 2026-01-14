export const deployKfApp = (
  bucket: string,
  region: string,
  kfRoleArn: string,
  certRoleArn: string,
  hostname: string,
  email: string,
) => {
  return {
    apiVersion: "argoproj.io/v1alpha1",
    kind: "Application",
    metadata: {
      name: "deploykf-app-of-apps",
      namespace: "argocd",
      labels: {
        "app.kubernetes.io/name": "deploykf-app-of-apps",
        "app.kubernetes.io/part-of": "deploykf",
      },
    },
    spec: {
      project: "default",
      source: {
        repoURL: "https://github.com/deployKF/deployKF.git",
        targetRevision: "v0.1.5",
        path: ".",
        plugin: {
          name: "deploykf",
          parameters: [
            {
              name: "source_version",
              string: "0.1.5",
            },
            {
              name: "values_files",
              array: ["./sample-values.yaml"],
            },
            {
              name: "values",
              string: `##
## This demonstrates how you might structure overrides for the 'sample-values.yaml' file.
## For a more comprehensive example, see the 'sample-values-overrides.yaml' in the main repo.
##
## Notes:
##  - YAML maps are RECURSIVELY merged across values files
##  - YAML lists are REPLACED in their entirety across values files
##  - Do NOT include empty/null sections, as this will remove ALL values from that section.
##    To include a section without overriding any values, set it to an empty map: \`{}\`
##

## --------------------------------------------------------------------------------
##                                      argocd
## --------------------------------------------------------------------------------
argocd:
  namespace: argocd
  project: default

## --------------------------------------------------------------------------------
##                                    kubernetes
## --------------------------------------------------------------------------------
kubernetes:
  {} # <-- REMOVE THIS, IF YOU INCLUDE VALUES UNDER THIS SECTION!

## --------------------------------------------------------------------------------
##                              deploykf-dependencies
## --------------------------------------------------------------------------------
deploykf_dependencies:

  ## --------------------------------------
  ##             cert-manager
  ## --------------------------------------
  cert_manager:
    images:
      kubectl:
        repository: docker.io/bitnamilegacy/kubectl
    controller:
      serviceAccount:
        annotations:
          eks.amazonaws.com/role-arn: ${certRoleArn}
    extraManifests:
      - |
        apiVersion: cert-manager.io/v1
        kind: ClusterIssuer
        metadata:
          name: letsencrypt-staging
        spec:
          acme:
            server: https://acme-staging-v02.api.letsencrypt.org/directory
            email: ${email}
            profile: tlsserver
            privateKeySecretRef:
              name: letsencrypt-staging
            solvers:
            - dns01:
                route53:
                  region: ${region}
                  role: ${certRoleArn}
                  auth:
                    kubernetes:
                      serviceAccountRef:
                        name: cert-manager
    clusterIssuer:
      enabled: false
      name: letsencrypt-staging

  ## --------------------------------------
  ##                 istio
  ## --------------------------------------
  istio:
    {} # <-- REMOVE THIS, IF YOU INCLUDE VALUES UNDER THIS SECTION!

  ## --------------------------------------
  ##                kyverno
  ## --------------------------------------
  kyverno:
    images:
      kubectl:
        repository: docker.io/bitnamilegacy/kubectl

## --------------------------------------------------------------------------------
##                                  deploykf-core
## --------------------------------------------------------------------------------
deploykf_core:

  ## --------------------------------------
  ##             deploykf-auth
  ## --------------------------------------
  deploykf_auth:
    images:
      kubectl:
        repository: docker.io/bitnamilegacy/kubectl

  ## --------------------------------------
  ##        deploykf-istio-gateway
  ## --------------------------------------
  deploykf_istio_gateway:
    gateway:
      hostname: ${hostname}
    gatewayService:
      name: "deploykf-gateway"
      type: "LoadBalancer"
      annotations:
        service.beta.kubernetes.io/aws-load-balancer-type: "external"
        service.beta.kubernetes.io/aws-load-balancer-nlb-target-type: "ip"
        service.beta.kubernetes.io/aws-load-balancer-scheme: "internet-facing"
      ports:
        http: 80
        https: 443

  ## --------------------------------------
  ##      deploykf-profiles-generator
  ## --------------------------------------
  deploykf_profiles_generator:
    profileDefaults:
      plugins:
        - kind: AwsIamForServiceAccount
          spec:
            awsIamRole: ${kfRoleArn}

## --------------------------------------------------------------------------------
##                                   deploykf-opt
## --------------------------------------------------------------------------------
deploykf_opt:

  ## --------------------------------------
  ##            deploykf-minio
  ## --------------------------------------
  deploykf_minio:
    enabled: false

  ## --------------------------------------
  ##            deploykf-mysql
  ## --------------------------------------
  deploykf_mysql:
    images:
      kubectl:
        repository: docker.io/bitnamilegacy/kubectl

## --------------------------------------------------------------------------------
##                                  kubeflow-tools
## --------------------------------------------------------------------------------
kubeflow_tools:

  ## --------------------------------------
  ##                 katib
  ## --------------------------------------
  katib:
    {} # <-- REMOVE THIS, IF YOU INCLUDE VALUES UNDER THIS SECTION!

  ## --------------------------------------
  ##               notebooks
  ## --------------------------------------
  notebooks:
    {} # <-- REMOVE THIS, IF YOU INCLUDE VALUES UNDER THIS SECTION!

  ## --------------------------------------
  ##               pipelines
  ## --------------------------------------
  pipelines:
    serviceAccounts:
      apiServer:
        annotations:
          eks.amazonaws.com/role-arn: ${kfRoleArn}
      frontend:
        annotations:
          eks.amazonaws.com/role-arn: ${kfRoleArn}

    bucket:
      name: ${bucket}
      region: ${region}

    objectStore:
      useExternal: true
      host: "s3.${region}.amazonaws.com"
      useSSL: true
      auth:
        fromEnv: true
kubeflow_dependencies:
  kubeflow_argo_workflows:
    controller:
      serviceAccount:
        annotations:
          eks.amazonaws.com/role-arn: ${kfRoleArn}
    server:
      serviceAccount:
        annotations:
          eks.amazonaws.com/role-arn: ${kfRoleArn}`,
            },
          ],
        },
      },
      destination: {
        server: "https://kubernetes.default.svc",
        namespace: "argocd",
      },
    },
  };
};

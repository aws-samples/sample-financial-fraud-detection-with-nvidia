# Data Folder

This folder holds local copies of datasets used for the Financial Fraud Detection project. All contents are gitignored.

## TabFormer Dataset

The primary dataset is IBM's **TabFormer** (`card_transaction.v1.csv`) — a synthetic credit card transaction dataset with ~24M records across 5,000 cardholders and 1,000 merchants.

For download instructions, see [notebooks/extra/download.md](../notebooks/extra/download.md).

After downloading, upload to S3 for pipeline use:

```bash
# Run 'make info' to find your actual bucket name
aws s3 cp card_transaction.v1.csv \
  s3://fraud-detection-<account>-sm/data/TabFormer/raw/ \
  --profile <your-aws-profile>
```
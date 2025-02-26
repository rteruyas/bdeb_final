#!/usr/bin/env python3
import os
import boto3
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model_from_s3():
    """Download the model file from S3"""
    s3_model_path = os.environ.get('S3_MODEL_PATH', 'bdeb-model-store/prod/model.pkl')
    local_model_path = os.environ.get('MODEL_PATH', '/app/model.pkl')
    
    if not s3_model_path:
        logger.error("S3_MODEL_PATH environment variable not set")
        return False
    
    try:
        # Parse S3 path
        bucket_name, key = parse_s3_path(s3_model_path)
        
        # Create S3 client
        s3_client = boto3.client('s3')
        
        logger.info(f"Downloading model from s3://{bucket_name}/{key} to {local_model_path}")
        
        # Download the file
        s3_client.download_file(bucket_name, key, local_model_path)
        
        logger.info("Model downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return False

def parse_s3_path(s3_path):
    """Parse S3 path into bucket and key components"""
    if s3_path.startswith('s3://'):
        s3_path = s3_path[5:]
    
    parts = s3_path.split('/', 1)
    if len(parts) == 2:
        bucket, key = parts
    else:
        bucket = parts[0]
        key = ''
    
    return bucket, key

if __name__ == "__main__":
    success = download_model_from_s3()
    if not success:
        logger.warning("Failed to download model. The API will start without a model loaded.")

import os
import sagemaker
from sagemaker.tensorflow import TensorFlow

sagemaker_session = sagemaker.Session()

role = "arn:aws:iam::335727716642:role/service-role/AmazonSageMaker-ExecutionRole-20200205T232539"
local_instance_type = "local"
remote_instance_type = "ml.t3.large"

source_dir = os.getcwd()
local_data_dir = os.path.join(os.getcwd(), "..")
remote_data_dir = "aiops-zhenqi/data"

estimator = TensorFlow(entry_point="sentiment_training.py",
                       base_job_name='train-cnn',
                       source_dir=source_dir,
                       role=role,
                       framework_version="1.14.0",
                       py_version="py3",
                       hyperparameters={},
                       train_instance_count=1,
                       train_instance_type=remote_instance_type)

local_inputs = {"train" : "file://" + local_data_dir + "/training/run-1582145830794-part-r-00000", 
                "validation" : "file://" + local_data_dir + "/validation/run-1582144896830-part-r-00000" , 
                "eval" : "file://" + local_data_dir + "/eval/run-1582146893400-part-r-00000"
                }

remote_inputs = {"train" : "s3://" + remote_data_dir + "/run-1582145830794-part-r-00000", 
                "validation" : "s3://" + remote_data_dir + "/run-1582144896830-part-r-00000", 
                "eval" : "s3://" + remote_data_dir + "/run-1582146893400-part-r-00000"
                }

estimator.fit(remote_inputs)

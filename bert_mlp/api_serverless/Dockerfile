FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt ${LAMBDA_TASK_ROOT}

# will install the torch dependencies manually so remove
# their entries from the requirements file;
RUN sed -i 's/torch.*//' ${LAMBDA_TASK_ROOT}/requirements.txt && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt

# copy necessary files to /var/task (aliased by LAMBDA_TASK_ROOT)
COPY saved_models/mlp_agnews.ckpt bert_mlp/api_serverless/inference.py bert_mlp/utils.py ${LAMBDA_TASK_ROOT}
CMD ["inference.handler"]
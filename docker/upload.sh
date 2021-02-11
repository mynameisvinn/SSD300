# upload docker image to aws ecr.
aws ecr get-login-password --region us-east-1 --profile kells | docker login --username AWS --password-stdin 553333793150.dkr.ecr.us-east-1.amazonaws.com

# retag.
image_name=ssd300-train-eval
ver=1.0.0
docker tag $image_name:$ver 553333793150.dkr.ecr.us-east-1.amazonaws.com/$image_name:$ver

# push.
docker push 553333793150.dkr.ecr.us-east-1.amazonaws.com/$image_name:$ver

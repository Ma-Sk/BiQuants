Login:
    
    aws ecr get-login --no-include-email --region us-east-1
    
Docker Build
- base:

    
    docker build -t 946988025622.dkr.ecr.us-east-1.amazonaws.com/base-look-alike -f ./base.Dockerfile .
    docker push 946988025622.dkr.ecr.us-east-1.amazonaws.com/base-look-alike

- dev:

    
    docker build -t 946988025622.dkr.ecr.us-east-1.amazonaws.com/look-alike:dev .
    docker push 946988025622.dkr.ecr.us-east-1.amazonaws.com/look-alike:dev

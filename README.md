# RetroTide

This python library designs PKSs automatically from chemical structures.

## Build image:
`docker build -t retrotide .`

## Run container
### Open bash:
`docker run -it --volume "${PWD}:/app" --workdir /app retrotide bash`

### Run jupyter notebooks:
`docker run -it --volume "${PWD}:/app" --workdir /app -p 8888:8888 retrotide jupyter notebook --no-browser --ip=0.0.0.0 --allow-root`
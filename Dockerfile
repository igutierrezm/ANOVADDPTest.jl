FROM julia:1.6.1
RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get -y install r-base

# Build
# docker build -t anovaddptest .

# Run
# sudo docker run --rm -di anovaddptest
# BrainProweredProduction
The production repo for the BrainPowered Project


* in the same directory as the dockerfile, create a directory on your local machine to store model
mkdir -p /model/

* build container
docker build . -t <name>

* run container
docker run -v /path/on/host:/data <image_name>
  (likely: docker run -v /:/data <name> )
# BrainProweredProduction
The production repo for the BrainPowered Project


* create a docker volume
docker volume create trainedBPModel

* build container
docker build . -t <name>

* run container
docker run -v /path/on/host/trainedBPModel:/data <name>
  (likely: docker run -v trainedBPModel:/data <name> )

* get location trained model:
docker volume inspect trainedBPModel
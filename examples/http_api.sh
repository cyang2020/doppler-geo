#!/bin/sh
# Start the server (needs: pip install 'doppler-geo[server]'):
#   dopgeo serve --port 8000
# Then, from another shell:
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/locate \
  -F file=@recording.wav \
  -F date=2026-09-08 -F tz=America/New_York \
  -F center_lat=40.8 -F center_lon=-73.4 -F radius_km=80

## AIS data based ML model to more accurately estimate ETA

### Notice
> Ships without /data! Not sure if we're licensed to redistribute these files. Acquire them on your own.

### Where to get /data
- Coastline dataset:
    - src: https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-coastline/
    - dst: /data/ne_10m_coastline/
- AIS dataset (Awake):
    - src: redacted
    - dst: /data/mission-voyage-examples/

### Setup
```bash
# clone repo
git clone https://github.com/Magnuswlange/port_congestion_proj_ml.git
cd port_congestion_proj_ml

# create venv, activate and install requirements
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt

# install jupyter kernel for .ipynb
python -m ipykernel install --user --name=port_congestion --display-name="Port Congestion (venv)"
```

# Decentralized_aml_demonstrator


# Quickstart(modeling only)
## 1） Quickstart (modeling only)

Assumes data already exists under:
- `data/raw/<bank>`

By default, bank names come from `config.yaml` (currently: `bank_s`, `bank_m`, `bank_l`).
```bash

cd /path/to/decentralized_aml_demonstrator/ # cd /home/admin_ml/Jackson/projects/aml/decentralized_aml_demonstrator/

deactivate # optional, deactivate AMLSim venv if active

pip install -r requirements.txt
source .venv/bin/activate

python scripts/02_process_data.py
python scripts/03_train_local_baseline.py
python scripts/04_federated_round.py
python scripts/05_evaluate.py
```
## 2）（Optional） Reproduce data with AMLSim


### 2.1 One-time AMLSim setup
0. Set AMLSim path

    ```bash
    export AMLSIM_DIR=$HOME/projects/AMLSim
    ```
   Example:
    ```bash
    export AMLSIM_DIR=/home/admin_ml/Jackson/projects/aml/amlsim/AMLSim 
    ```

1. System dependencies (Ubuntu):
    ```bash
    sudo apt-get update
    sudo apt-get install -y maven graphviz graphviz-dev pkg-config
    ```

2. Verify Java:
    ```bash
    java -version
    javac -version
    ```

3. Install Python 3.7 via pyenv(is missing)
    ```bash
    pyenv install 3.7.17
    ```

4. Create AMLSim venv:
    ```bash
    cd "$AMLSIM_DIR"
    pyenv local 3.7.17

    python -m venv .venv
    source .venv/bin/activate
    pip install -U pip
    pip install -r requirements.txt
    pip install pyyaml
    deactivate
    ```

5. Java build: install external MASON jar(required)
    AMLSim depends on mason:mason:20 which is not in Maven Central, so it must be installed locally.
    ```bash
    cd "$AMLSIM_DIR"

    # download mason jar
    wget -O jars/mason.20.jar https://cs.gmu.edu/~eclab/projects/mason/mason.20.jar

    # install into local maven repo
    mvn install:install-file \
    -Dfile=jars/mason.20.jar \
    -DgroupId=mason \
    -DartifactId=mason \
    -Dversion=20 \
    -Dpackaging=jar
    ```

6. Build AMLSim jar
    Important: use bash (not sh) for AMLSim scripts
    ```bash
    cd "$AMLSIM_DIR"
    bash scripts/build_AMLSim.sh
    ```

### 2.2 Generate data with AMLSim
AMLSim is an external dependency (cloned outside this repo).

0. Set AMLSim path

    ```bash
    export AMLSIM_DIR=$HOME/projects/AMLSim
    ```
   Example:
    ```bash
    export AMLSIM_DIR=/home/admin_ml/Jackson/projects/aml/amlsim/AMLSim 
    ```

1. Use AMLSim's Python environment to run data generation script--01_generate_data.py, which is in this repo.

    ```bash
    source "$AMLSIM_DIR/.venv/bin/activate"

    cd /path/to/decentralized_aml_demonstrator/ # cd /home/admin_ml/Jackson/projects/aml/decentralized_aml_demonstrator/
    python scripts/01_generate_data.py
    ```
    After it completes, generated files will be copied into decentralized_aml_demonstrator/data/raw folder. Remember to deactivate AMLSim venv after use:
    ```bash
    deactivate
    ```



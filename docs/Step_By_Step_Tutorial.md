Step by Step Tutorial - Hotel Reservation - MLOps End to End
============================================================

Author: Cao Tri DO (<caotri.do88@gmail.com>)

Part 0: Setup your machine
----------------------------------------------------
1. Install git: `sudo apt install git-all`
2. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`

You can also install Taskfile. Go the home:
```sql
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d
ls -l ~/bin/task
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Part 1: Play as the Architect - Configure your infra
----------------------------------------------------

1. Open a new Databricks Free Edition Account

https://login.databricks.com/signup?tuuid=174967b9-5bc8-46cf-99ed-0dd2e521b2b1&dbx_source=direct&intent=SIGN_UP&sisu_state=eyJsZWdhbFRleHRTZWVuIjp7Ii9zaWdudXAiOnsicHJpdmFjeSI6dHJ1ZSwiY29ycG9yYXRlRW1haWxTaGFyaW5nIjp0cnVlfX19

2. Create an access token 

Settings --> User --> Developer --> Access tokens --> Manage Generate new token
Note the token

3. Create 3 catalogs in Databricks mlops_acc, mlops_dev and mlops_prd
```sql
-- Create Development Catalog
CREATE CATALOG IF NOT EXISTS mlops_dev 
COMMENT 'MLOps Development Environment';

-- Create Acceptance (Staging) Catalog
CREATE CATALOG IF NOT EXISTS mlops_acc 
COMMENT 'MLOps Acceptance/Staging Environment';

-- Create Production Catalog
CREATE CATALOG IF NOT EXISTS mlops_prd 
COMMENT 'MLOps Production Environment';
```

Part 2: Play as a Data Scientist - Launch the project
-----------------------------------------------------
0. Fork and clone the repo
```bash
git clone git@github.com:malganis35/hotel-reservation-databricks-free.git
```

1. Install the Databricks CLI
```bash
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sudo sh
```

2. Configure the Databricks profiles
```bash
databricks auth login --host https://dbc-c36d09ec-dbbe.cloud.databricks.com/
```

Note: You can check your configuration
```bash
cat ~/.databrickscfg
```

3. Create a .env file
```bash
PROFILE=dev-free
DATABRICKS_HOST=https://dbc-c36d09ec-dbbe.cloud.databricks.com
DATABRICKS_TOKEN=xxxxxxxxxxxxx
DATABRICKS_COMPUTE=serverless
```

4. Make a global research on VSCode and replace 

```bash
DATABRICKS_HOST=https://dbc-c36d09ec-dbbe.cloud.databricks.com
by
DATABRICKS_HOST=xxxxxxxxxxxxx
with your own host
```

5. Test that everything works in the command line
```bash
uv run ./notebooks/demo.py
```

6. Open VSCode on the project folder, install the databricks Extension and configure the databricks extension

7. Open the notebook ./notebooks/demo.py 
- select the right kernel (hotel-reservation)
- run it from the interactive console in VSCode
- everything will run with the databricks cluster

8. Upload the raw data from Kaggle in to the Unity Catalog
```bash
uv run notebooks/utils/run_upload_data.py --env dev
```

9. Process the raw data into a Table in the Unity Catalog
```
uv run ./notebooks/process_data.py --branch dev
```

10. Create a workspace to share the experiments
```bash
uv run ./notebooks/utils/run_create_mlflow_workspace.py 
```

10. Train a Basic LR ML Model into MLFlow and register the champion model into Unity Catalog
```bash
uv run ./notebooks/train_register_basic_model.py --branch dev --git_sha 1234 --job_run_id 1234
```

12. Test the latest model from registry on 10 predictions
```bash
uv run ./notebooks/predict_basic_model.py --branch dev --git_sha 1234 --job_run_id 1234
```

11. Deploy the champion ML Model as a model serving endpoint in Databricks (API) 
```bash
uv run ./notebooks/deploy_basic_model_serving.py --branch dev
```
Note: it might take up to 7 min to create the endpoint

13. Alternatively, you can also test the endpoint from your command line
Go to the file ./tests/functional/example.http and copy the 2nd curl request
```bash
cd ./tests/functional/
curl \
  -X POST \
  -H "Authorization: Bearer $(databricks auth token --host https://dbc-c36d09ec-dbbe.cloud.databricks.com | jq -r '.access_token')" \
  -H "Content-Type: application/json" \
  -d @data.json \
  https://dbc-c36d09ec-dbbe.cloud.databricks.com/serving-endpoints/hotel-reservation-basic-model-serving-db/invocations
```

14. Note: In the premium edition, you can activate the AI Gateaway inference Table to log the API call on the endpoint and activate the Lakehouse Monitoring


Part 3: Play as the ML Engineer - Deploy & Automate the project
---------------------------------------------------------------

1. Validate the DAB 
```bash
databricks bundle validate -t dev
```

2. Deploy the DAB on Databricks
```bash
databricks bundle deploy -t dev
```

3. Run a workflow
```bash
databricks bundle run -t dev
```

Note: You can also run the workflow manually from the Databricks "Jobs & Pipelines" interface

4. Configure your Github Repo to automatically deploy in acc and prd environment



Part 4: Play the Data Analyst - Deploy & Run the Streamlit Interface
--------------------------------------------------------------------

1. Change in `apps/app.py` the url of your endpoint:

serving_endpoint = "https://dbc-c36d09ec-dbbe.cloud.databricks.com/serving-endpoints/hotel-reservation-basic-model-serving-db/invocations"

2. Access to the Apps

Go to Compute --> Apps --> Select the Apps --> Start

Note: it might take 2 to 3 minutes to start

3. Select the path folder 

```bash
/Workspace/Users/ct_do@msn.com/.bundle/dev/hotel-reservation-caotrido/files/app
```

4. Click on the URL

5. Make Predictions

Be careful, the app points out on the dev API endpoint 

Note: Only Prediction Demo will work on Free Edition because AI Gateaway Inference Log is not available anymore on Free Edition

Bonus: clean up the ressources
------------------------------

1. Destroy the DAB

```bash
databricks bundle destroy
```

2. Delete all MLFlow experiments
```bash
uv run ./notebooks/utils/run_cleanup_mlflow_experiments.py
```

3. Clean up all the data in the catalog
```bash
uv run ./notebooks/utils/run_cleanup_data.py
```

4. Delete the Catalog
```sql
DROP SCHEMA IF EXISTS mlops_dev.hotel_operations CASCADE;
DROP SCHEMA IF EXISTS mlops_acc.hotel_operations CASCADE;
DROP SCHEMA IF EXISTS mlops_prd.hotel_operations CASCADE;
```

5. Delete manually the serving endpoint
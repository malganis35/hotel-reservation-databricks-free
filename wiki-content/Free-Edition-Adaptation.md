# All elements to adapt for the Databricks Free Edition

## Inference Table

Inference table are not part of the functionalities in the Databricks Free Edition

Replace this:

```python
serving.deploy_or_update_serving_endpoint(
            version=entity_version_latest_ready,
            enable_inference_tables=True,
        )
```

by this

```python
serving.deploy_or_update_serving_endpoint(
            version=entity_version_latest_ready,
            enable_inference_tables=False,
        )
```

## Number of serving endpoints

Our tests has shown that it is not possible to have more than 2 serving endpoints exposed.

If you create more, you will see an error like this one:

```bash
TimeoutError: Timed out after 0:05:00
[Trace ID: 00-228885a185ea176ad330ec85d82fac76-a90a04c73d05feff-00]
---------------------------------------------------------------------------
ResourceExhausted                         Traceback (most recent call last)
File /local_disk0/.ephemeral_nfs/envs/pythonEnv-22440b02-15e8-4777-b366-3940bd4df422/lib/python3.12/site-packages/databricks/sdk/retries.py:36, in retried.<locals>.decorator.<locals>.wrapper(*args, **kwargs)
     35 try:
---> 36     return func(*args, **kwargs)
     37 except Exception as err:

File /local_disk0/.ephemeral_nfs/envs/pythonEnv-22440b02-15e8-4777-b366-3940bd4df422/lib/python3.12/site-packages/databricks/sdk/_base_client.py:298, in _BaseClient._perform(self, method, url, query, headers, body, raw, files, data, auth)
    297 if error is not None:
--> 298     raise error from None
    300 return response

ResourceExhausted: You've hit the limit for endpoints for free usage. Stop or delete existing endpoints to free up capacity.
```

We tried to stop the endpoint but the problems remains the same. The only way is to delete other endpoints.

In our case, the endpoints are deployed quite rapidly (about 10 min)

## Databricks Asset Bundle (DAB): Serverless Cluster Definition

### 1. Delete `cluster_id` definition in targets

As we are using serverless cluster in the Databrick Free Edition, you can't define `cluster_id` argument in DAB

```yml
targets:
  dev:
    default: true
    mode: development
    workspace:
      host: https://dbc-c36d09ec-dbbe.cloud.databricks.com
      root_path: /Workspace/Users/${workspace.current_user.userName}/.bundle/${bundle.target}/${bundle.name}
    variables:
      schedule_pause_status: PAUSED
      is_test: 0
      env: dev
    artifacts:
      default:
        type: whl
        build: uv build
        path: .
        dynamic_version: True
```

So the solution is just to delete `cluster_id`

### 2. Delete `job_clusters` in resources/jobs

When defining a job pipeline, in general you define the `job_clusters`configuration (with the spark version, etc.). This is not possible in Free Edition

```yml
resources:
  jobs:

    initial_training:
      name: ${bundle.name}-initial-training
      tags:
        project_name: "hotel-reservation-caotrido"
        job_type: "initial_training"
        environment: "${var.env}"
        model_name: "basic_hotel_reservation_model"
        finops_cost_center: "mlops-cohort4"
        purpose: "ml-training"
      job_clusters:
        - job_cluster_key: "hotel-reservation-cluster"
          new_cluster:
            spark_version: "16.4.x-scala2.12"
            data_security_mode: "SINGLE_USER"
            node_type_id: "r3.xlarge"
            driver_node_type_id: "r3.xlarge"
            autoscale:
              min_workers: 1
              max_workers: 1
            spark_env_vars:
              "TOKEN_STATUS_CHECK": "{{secrets/mlops_course/git_token_status_check}}"
```

You will need to delete the part `job_clusters`

### 3. Add `environments` section in resources/jobs

Add the following `environments` section in the resources/jobs section. We recommend to use environment_version = 3

```yml
resources:
  jobs:

    initial_training:
        name: ${bundle.name}-initial-training
        
        environments:
            - environment_key: default
            spec:
                environment_version: "3"
                dependencies:
                - ../dist/*.whl
```

### 3. Replace `existing_cluster_id` by `environment_key` in resources/jobs/tasks

In the tasks of your jobs of the ressources, `existing_cluster_id` will need to be deleted because we cannot specify a cluster.

Instead, you will have to add `environment_key: default` so that it will use the environment defined above.

```yml
resources:
  jobs:

    initial_training:
        name: ${bundle.name}-initial-training
        
        environments:
            - environment_key: default
            spec:
                environment_version: "3"
                dependencies:
                - ../dist/*.whl
        tasks:
        - task_key: "initial_processing"
            environment_key: default
            spark_python_task:
                python_file: "../scripts/00.process_initial_data.py"
```

Careful, when you have a conditional_task in a task, do not add `environment_key: default`

```yml
resources:
  jobs:

    initial_training:
        name: ${bundle.name}-initial-training
        
        environments:
            - environment_key: default
            spec:
                environment_version: "3"
                dependencies:
                - ../dist/*.whl
        tasks:
        - task_key: post_commit_status_required
          condition_task:
            op: "EQUAL_TO"
            left: "${var.is_test}"
            right: "1"
          depends_on:
            - task_key: "deploy_model"
```

4. Delete `libraries` section in resources/jobs/tasks

In the premium edition, we add the library section to define the depedency of the run such as the .whl for example of the package.

In the Free Edition, as we defined the `environments`, this section is not needed and can be deleted:

```yml
resources:
  jobs:

    initial_training:
        name: ${bundle.name}-initial-training
        
        environments:
            - environment_key: default
            spec:
                environment_version: "3"
                dependencies:
                - ../dist/*.whl
        tasks:
        - task_key: "initial_processing"
            environment_key: default
            spark_python_task:
                python_file: "../scripts/00.process_initial_data.py"

            libraries:
                - whl: ../dist/*.whl
```

So just delete the section:

```yml
    libraries:
        - whl: ../dist/*.whl
```

## Online Table

Online Table is not available for the Databricks Free Edition
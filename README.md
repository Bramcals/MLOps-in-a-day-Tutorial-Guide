# Data & AI Tech Immersion Workshop – Product Review Guide and Lab Instructions

## Scenario overview

In this experience you will learn how Contoso Auto can use MLOps to formalize the process of training and deploying new models using a DevOps approach.
The Contoso Corporation is a fictional but representative global manufacturing conglomerate. In this tutorial we will use a train script that has already been built to create a model that will predict whether a car is a compliant car, i.e., whether it meets tightening government regulations for low-emission vehicles. The dataset contains information about the condition of car components, type of material and its manufacturing year.

## Technology overview

Azure Machine Learning uses a Machine Learning Operations (MLOps) approach, which improves the quality and consistency of your machine learning solutions. Azure Machine Learning Service provides the following MLOps capabilities:

- Integration with Azure Pipelines. Define continuous integration and deployment workflows for your models.
- A model registry that maintains multiple versions of your trained models.
- Model validation. Automatically validate your trained models and select the optimal configuration for deploying them into production.
- Deploy your models as a web service in the cloud, locally, or to IoT Edge devices.
- Monitor your deployed model's performance, so you can drive improvements in the next version of the model.

## AI, Experience - MLOps with Azure Machine Learning and Azure DevOps

- [Data &amp; AI Tech Immersion Workshop – Product Review Guide and Lab Instructions](#data-amp-ai-tech-immersion-workshop-%e2%80%93-product-review-guide-and-lab-instructions)
  - [AI, Experience 6 - MLOps with Azure Machine Learning and Azure DevOps](#ai-experience-6---mlops-with-azure-machine-learning-and-azure-devops)
  - [Technology overview](#technology-overview)
  - [Scenario overview](#scenario-overview)
  - [Prerequisite: Create a resource group](#Prerequisite-Create-a-resource-group)
  - [Exercise 1: Setup New Project in Azure DevOps](#exercise-1-setup-new-project-in-azure-devops)
    - [Task 1: Create New Project](#task-1-create-new-project)
    - [Task 2: Import Quickstart code from a GitHub Repo](#task-2-import-quickstart-code-from-a-github-repo)
    - [Task 3: Create a variable group](#task-3-create-a-variable-group)
    - [Task 4: Create new Service Connection with Resource Group](#task-4-create-new-service-connection-with-resource-group)
  - [Exercise 2: Setup and Run the CI Pipeline](#exercise-2-setup-and-run-the-CI-pipeline)
    - [Task 1: Setup the CI Pipeline](#task-1-setup-the-CI-pipeline)
    - [Task 2: Run the CI pipeline](#task-2-run-the-CI-pipeline)
    - [Task 3: Review output CI pipeline](#task-3-review-output-of-CI-pipeline)
  - [Exercise 3: Setup and Run the IAC Pipeline](#exercise-3-setup-and-run-the-IAC-pipeline)
    - [Task 1: Setup the IAC Pipeline](#task-1-setup-the-IAC-pipeline)
    - [Task 2: Run the IAC pipeline](#task-2-run-the-IAC-pipeline)
    - [Task 3: Review output of IAC Pipeline](#task-3-review-output-of-IAC-Pipeline)
    - [Task 4: Create new Service Connection with Azure Machine Learning](#task-4-create-new-service-connection-with-azure-machine-learning)
  - [Exercise 4: Setup and Run the Train Pipeline](#exercise-4-setup-and-run-the-train-pipeline)
    - [Task 1: Setup Train Pipeline](#task-1-setup-train-pipeline)
    - [Task 2: Run the Train Pipeline](#task-2-run-the-train-pipeline)
    - [Task 3: Review Train Artifacts](#task-3-review-train-artifacts)
    - [Task 4: Review Train Outputs](#task-4-review-train-outputs)
  - [Exercise 5: Setup the Release Pipeline](#exercise-5-setup-the-release-pipeline)
    - [Task 1: Create an Empty Job](#task-1-create-an-empty-job)
    - [Task 2: Add Build Artifact](#task-2-add-build-artifact)
    - [Task 3: Add Variables to Deploy &amp; Test stage](#task-3-add-variables-to-deploy-amp-test-stage)
    - [Task 4: Setup Agent Pool for Deploy &amp; Test stage](#task-4-setup-agent-pool-for-deploy-amp-test-stage)
    - [Task 5: Add Use Python Version task](#task-5-add-use-python-version-task)
    - [Task 6: Add Install Requirements task](#task-6-add-install-requirements-task)
    - [Task 7: Add Deploy &amp; Test Webservice task](#task-7-add-deploy-amp-test-webservice-task)
    - [Task 8: Define Deployment Trigger](#task-8-define-deployment-trigger)
    - [Task 9: Enable Continuous Deployment Trigger](#task-9-enable-continuous-deployment-trigger)
    - [Task 10: Save the Release Pipeline](#task-10-save-the-release-pipeline)
  - [Exercise 6: Run the Release Pipeline](#Exercise-6-run-the-release-pipeline)
    - [Task 1: Start Release pipeline](#task-1-start-release-pipeline)
    - [Task 2: Monitor Release Pipeline](#task-2-monitor-release-pipeline)
  - [Exercise 7: Pull a request from the model via Postman](#exercise-7-pull-request)
    - [Task 1: Get URI and Primary key of model](#task-1-get-uri)
    - [Task 2: Pull a request from the deployed model via Postman](#task-2-pull-request-in-postman)
  - [Wrap-up](#wrap-up)
  - [Deletion Exercise: Delete resource in the Azure Portal](#Deletion-exercise-delete-resource-in-the-Azure-Portal)
  - [Take-Home Exercise: Test train and Release Pipelines](#Take-Home-Exercise-Test-train-and-Release-Pipelines)
  - [Additional resources and more information](#additional-resources-and-more-information)


## Prerequisite: resource group

A resource group is a container for resources (Azure services). It is equivalent to a folder that contains files.

__Note__: We have already created a resource group for you, named: `MLOPS-RG-X` (replace `X` with your student number). Continue with Exercise 1.

To create a resource group:
1. Go to the Azure Portal: portal.azure.com
2. Go to home and click on Resource Groups
3. Click on +Create
4. Choose your Subscription
5. Choose a name for the Resource Group (no longer than 10 characters!). For example `MLOPS-{your initials}`
6. Set region to `(Europe) West Europe`
7. Click on -> Review + Create -> Create

## Exercise 1: Setup New Project in Azure DevOps

In this exercise you will set up a project in DevOps, import a repository that we have build for you and you will create a connection with the Azure Machine Learning service.

Duration: 20 minutes

### Task 1: Create New Project (Project has already been set up for you, continue with task 2)

1. Sign in to [Azure DevOps](http://dev.azure.com).

2. Create your own organization or use an existing one. As long as it's your own organization. Choose your own name when creating a new organization

3. Select **New project**.

   ![Create new project in Azure DevOPs.](media/devops-project-01.png "Create new project")

4. Provide Project Name: `mlops-quickstart` and select **Create**.

   ![Provide project name in the create new project dialog and then select create.](media/devops-project-02.png "Create New Project Dialog")

### Task 2: Import Quickstart code from a GitHub Repo

In this task you import a repository from GitHub. This repository mostly consists of Python files and several YAML files. The Python files will perform the Data Science steps such as training, evaluating and deploying a model. The YAML files, are used to set up the pipelines in DevOps and determine which Python files to execute in which order.

1. Within the new project:

   a. Select **Repos** from left navigation bar.

   b. Select **Import** from the content section.

   ![Import Quickstart code from a GitHub Repo.](media/devops-project-03.png "Azure DevOps Repos")

2. Provide the following GitHub URL: TODO `https://github.com/Bramcals/MLOps-starter.git` and select **Import**. This should import the code required for the quickstart.

   ![Provide the above GitHub URL and select import to import the source code.](media/devops-project-04.png "Import a Git repository dialog")

### Task 3: Create a variable group

1. Select **Pipelines**, select **Library** and select **+ Variable group**

   ![Go to create variable group section](media/variable-group.png "Create variable group")

2. Name the variable group; `quickstart-variablegroup` and add the following variables;

   a. `LOCATION` = `westeurope`

   b. `RESOURCE_GROUP` = `MLOPS-RG-X` (replace `X` with your student number)

   c. `WORKSPACE_NAME` = `MLOPS-X-AML` (replace `X` with your student number)

   d. `BASE_NAME` = workspace name in lower case letter and no special characters: `mlopsxaml` (replace `x` with your student number)

   ![Add variables to variable group section](media/variable-group-vqd.png "adding variables to variable group")

3. Select `Save`

### Task 4: Create new Service Connection with Resource Group 

Once the correct resource group and Azure Machine Learning name have been provided, a connection can be made. First we will create a connection with the resource group.

1. From the left navigation select **Project settings** and then select **Service connections**.

   ![Navigate to Project Settings, Service connections section.](media/devops-build-pipeline-03.png "Service Connections")

2. Select **Create service connection**, select **Azure Resource Manager**, and then select **Next**.

   ![Select Create Service Connection, Azure Resource Manager.](media/devops-build-pipeline-04.png "Azure Resource Manager")

3. Select **Service principal (automatic)** and then select **Next**.

   ![Select Service principal (automatic), and then select Next.](media/devops-build-pipeline-05.png "Service principal authentication")

4. Provide the following information for the 'New Azure service connection' and then select **Save**:

   a. Scope Level: `Subscription`

   > **Note**: If you are unable to select your **Resource group**, do the following steps:

   - Quit the `Service connection` dialog
   - Make sure Cookies are allowed by the browser
   - Refresh or reload the web browser
   - Click on create new connection
   - Perform steps 1 - 3 again
   - In step 4, change the `Scope level` to **Subscription**
   - Then a Microsoft Login windows should appear. Provide your login details
   - Now, you should be able to select your resource group
   - Name your service connection: `quick-starts-sc-rg`
   - Grant access permission to all pipelines

   b. Subscription: `VQD Data Science`

   c. Resource group: This value should match the value you just provided in the library as a variable: `MLOPS-RG-X` (replace `X` with your student number)

   d. Service connection name: `quick-starts-sc-rg`

   e. Grant access permission to all pipelines: this checkbox must be selected.

   f. Select `Save`

   ![Provide connection name, Azure Resource Group, Machine Learning Workspace, and then select Save. The resource group and machine learning workspace must match the value you provided in the YAML file.](media/sc-rg.png "Add an Azure Resource Manager service")


## Exercise 2: Setup and Run the CI Pipeline

In this exercise, the CI will be built. In this pipeline a code quality check will be performed on all Python files in the repository. Unit tests can also be performed in this pipeline. In unit testing you break down the functionality of your program into discrete testable behaviors that you can test as individual units. However, for the sake of this tutorial, we will only do a code quality check.

1. From left navigation select **Pipelines, Pipelines** and then select **Create pipeline**.

   ![Navigate to Pipelines, Pipelines, and then select Create pipeline](media/devops-build-pipeline-07.png "Create Build Pipeline")

2. Select **Azure Repos Git** as your code repository.

   ![Select your code repository source for your new build pipeline.](media/devops-build-pipeline-08.png "Repository Source")

3. Select **mlops-quickstart** as your repository.

   ![Select mlops-quickstart as your repository.](media/devops-build-pipeline-09.png "Select Repository")

4. Select **Existing Azure Pipelines YAML file**, select **/environment_setup/CI-pipeline.yml** as your path and select continue

   ![Ci pipeline.](media/CI-Pipeline-path.png "Select CI path")

5. Review your pipeline YAML

   a. Note that Python is installed to perform this step

   b. The main step in this pipeline is the code quality check. It is also possible to create a report of this step. However, this is left out of scope.

### Task 2: Run the CI Pipeline

1. Before running the pipeline, let us first give the pipeline a meaningful name. Select the arrow next to the run button.

   ![Select yaml file as your setup.](media/save-CI-pipeline.png "Select save")

2. Select **Save**

3. Select the settings button next to the **Run Pipeline** button and select **Rename/move**

   ![Select yaml file as your setup.](media/settings-save-rename.png "Select save")

4. Rename the pipeline to **CI-Pipeline**

5. Select **Run Pipeline** and press **Run**

### Task 3: Review output of CI pipeline

1. Select **Job** to view the current progress in the pipeline execution run

   ![Select job.](media/Select-job.png "Select job")

2. Review the steps

3. Review the code quality check. In the check it can be seen that several packages are imported but never used. The check also indicates missing or redundant white lines/space and lines exceeding the character limit. With this check, we can adjust our code and make it more pythonic to increase standardization and manageability of work.

   ![Check code quality.](media/Code-quality-check.png "Check code quality")

## Exercise 3: Setup and Run the IAC Pipeline

In this exercise, the IAC pipeline will be built. This pipeline will create the resources or update if they already exist.

### Task 1: Setup the IAC Pipeline

1. From left navigation select **Pipelines, Pipelines** and then select **New pipeline**.

2. Select **Azure Repos Git** as your code repository.

3. Select **mlops-quickstart** as your repository.

4. Select **Existing Azure Pipelines YAML file**, select **/environment_setup/IAC-pipeline.yml** as your path and select continue

5. Review your pipeline YAML

   a. note that the YAML file is connected to your **quickstart-variablegroup**

   b. In this pipeline YAML, only one step is performed: **Azure Resource Group Deployment**. In this YAML file you can configure the pipeline to load your data into a storage account or attach computes to your resources.

### Task 2: Run the IAC Pipeline

1. Before running the pipeline, let us first give the pipeline a meaningful name. Select the arrow next to the run button.

   ![Select yaml file as your setup.](media/save-before-run.png "Select save")

2. Select **Save**

3. Select the settings button next to the **Run Pipeline** button and select **Rename/move**

   ![Select yaml file as your setup.](media/settings-save-rename.png "Select save")

4. Rename the pipeline to **IAC-Pipeline**

5. Select **Run Pipeline** and press **Run**

### Task 3: Review output of IAC Pipeline

1. Select **Job** to view the current progress in the pipeline execution run

   ![Select job.](media/Select-job.png "Select job")

2. Review the steps

3. Note that Deploy MLOps Resources requires most time.

   ![Review steps of IAC pipeline.](media/review-steps.png "Review steps")

### Task 4: Create new Service Connection with Azure Machine Learning

Now that the Azure Machine Learning (AML) resource is created, we will create a connection between DevOps and the AML resource.

1. Select **New service connection**.

   ![Navigate to Project Settings, Service connections section.](media/devops-build-sc-rg.png "Service Connections")

2. Select **Azure Resource Manager**, and then select **Next**.

   ![Select Create Service Connection, Azure Resource Manager.](media/devops-build-pipeline-04.png "Azure Resource Manager")

3. Select **Service principal (automatic)** and then select **Next**.

   ![Select Service principal (automatic), and then select Next.](media/devops-build-pipeline-05.png "Service principal authentication")

4. Provide the following information in the `New Azure service connection` dialog box and then select **Save**:

   a. Scope Level: `Machine Learning Workspace`

   > **Note**: If you are unable to select your **Machine Learning Workspace**, do the following steps:

   - Quit the `New Azure service connection` dialog
   - Make sure Cookies are allowed by the browser
   - Refresh or reload the web browser
   - Click on create new connection
   - Perform steps 1 - 3 again
   - In step 4, change the `Scope level` to **Subscription**
   - Then a Microsoft Login windows should appear. Provide your login details
   - Now, you should be able to select your resource group and AML workspace
   - Name your service connection: `quick-starts-sc-aml`
   - Grant access permission to all pipelines

   b. Subscription: `VQD Data Science`

   <!-- > **Note**: It might take up to 30 seconds for the **Subscription** dropdown to be populated with available subscriptions, depending on the number of different subscriptions your account has access to. -->

   c. Resource group: This value should match the value you just provided in the library as variable: `MLOPS-RG-X` (replace `X` with your student number) 

   d. Machine Learning Workspace: This value should match the value you just provided in the library as variable: `MLOPS-X-AML` (replace `X` with your student number)

   e. Service connection name: `quick-starts-sc-aml`

   f. Grant access permission to all pipelines: this checkbox must be selected.

   g. Select `Save`

   ![Provide connection name, Azure Resource Group, Machine Learning Workspace, and then select Save. The resource group and machine learning workspace must match the value you provided in the YAML file.](media/sc-aml.png "Add an Azure Resource Manager service")

   **Note**: If you successfully created the new service connection **go to Exercise 2**.


## Exercise 4: Setup and Run the Train Pipeline

In this exercise, the Train pipeline will be set up. A pipeline is attached to a repository that must contain a file with all the steps required to execute in the pipeline. In this tutorial, a YAML file is available that contains these steps. After setting up the pipeline, the pipeline can be executed. DevOps creates an agent that will perform the pipeline steps described in `train-pipeline.yml`. The first step in the pipeline, is to install Python. Then several packages are installed, needed to execute the Python files. Once CLI and AML have been set-up, the agent kicks off the master pipeline. In the master pipeline, first the model is trained and then evaluated. In the evaluation step, the accuracy of the model is compared with the accuracy of the currently deployed model. If the accuracy of the new model is higher or there is no model yet deployed, the new model will be deployed. If the accuracy is lower than the currently deployed model, the new model will not be deployed. Whether or not a model will be deployed, is saved in a eval_info.json file. This file, together with the model itself, are outputs of the Train pipeline and used in the deployment pipeline.

Duration: 25 minutes

### Task 1: Setup Train Pipeline

1. From left navigation select **Pipelines, Pipelines** and then select **New pipeline**.

2. Select **Azure Repos Git** as your code repository.

3. Select **mlops-quickstart** as your repository.

4. Select **Existing Azure Pipelines YAML file**, select **/environment_setup/Train-pipeline.yml** as your path and select continue

5. Review the YAML file.

   The train pipeline has four key steps:

   a. Attach folder to workspace and experiment. This command creates the `.azureml` subdirectory that contains a `config.json` file that is used to communicate with your Azure Machine Learning workspace. All subsequent steps rely on the `config.json` file to instantiate the workspace object.

   b. Create the AML Compute target to run your master pipeline for model training and model evaluation.

   c. Run the master pipeline. The master pipeline has two steps: (1) Train the machine learning model, and (2) Evaluate the trained machine learning model. These steps are submitted to AML. On AML we have just created a compute that will execute the two steps. The results of the two steps can be seen in the Azure Machine Learning workspace. In the training step, the model is trained with an XGboosting algorithm.

   The evaluation step evaluates if the new model performance is better than the currently deployed model. If the new model performance is improved, the evaluate step will create a new Image for deployment. The results of the evaluation step will be saved in a file called `eval_info.json` that will be made available for the release pipeline. You can review the code for the master pipeline and its steps in `aml_service/pipelines_master.py`, `scripts/train.py`, and `scripts/evaluate.py`.

   d. Publish the train artifacts. The `snapshot of the repository`, `config.json`, and `eval_info.json` files are published as train artifacts and thus can be made available for the release pipeline.

### Task 2: Run the Train Pipeline

1. Before running the pipeline, let us first give the pipeline a meaningful name. Select the arrow next to the run button.

2. Select **Save**

3. Select the settings button next to the **Run Pipeline** button and select **Rename/move**

   ![Select yaml file as your setup.](media/settings-save-rename.png "Select save")

4. Rename the pipeline to **Train-Pipeline**

5. Select **Run Pipeline** and select **Run** to start running your train pipeline.

6. Monitor the train run. The train pipeline, for the first run, will take around 10 minutes to run.

   ![Monitor your train pipeline. It will take around 20 minutes to run.](media/devops-build-pipeline-12.png "Monitor train Pipeline")

7. Select **Job** to monitor the detailed status of the train pipeline execution.

   ![Monitor the details of your train pipeline.](media/devops-build-pipeline-13.png "Monitor train Pipeline Details")

### Task 3: Review Train Artifacts

1. The train pipeline will publish an artifact named `devops-for-ai`. Go to Pipelines -> Click on the `Train-Pipeline` -> Select the latest run -> Select **Related, 1 Published** to review the artifact contents.

   ![Select Artifacts, 1 published to review the artifact contents.](media/screenshot-artifact.png "Build Artifacts")

2. Select **outputs, eval_info.json** and then select the download arrow. The file `eval_info.json` contains the output from the _model evaluation_ step which will later be used in the release pipeline to deploy the model. Click on the back arrow to return to the previous screen.

   ![Download output from the model evaluation step.](media/devops-build-pipeline-15.png "Download JSON file")

3. Open the `eval_info.json` in a JSON viewer or a text editor. The JSON output contains information such as the status of the evaluation step (`deploy_model`: _true_ or _false_), or the name and id of the created image (`image_name` and `image_id`) to deploy.

   ![Review information in the eval_info json file.](media/devops-build-pipeline-16.png "Eval Info JSON File")

### Task 4: Review Train Outputs

1. Log in to [Azure Machine Learning studio](https://ml.azure.com) either directly or via the [Azure Portal](https://portal.azure.com). Make sure you select the Azure Machine Learning workspace that you created from the notebook earlier. Open your **Models** section, and observe the versions of the registered model: `compliance-classifier`. The latest version is the one registered by the train pipeline you have run in the previous task.

   ![Review registered model in Azure Machine Learning studio.](media/devops-build-outputs-01.png "Registered Models in Azure Machine Learning studio")

2. Select the latest version of your model to review its properties. Notice the `build_number` tag which links the registered to model to the Azure DevOps build that generated it.

   ![Review registered model properties, notice Build_Number tag.](!media/../media/devops-build-outputs-02.png "Registered model details and Build_Number tag")

3. Open the **Datasets** tab and observe the versions of the registered dataset: `connected_car_components`. The latest version is the one registered by the train pipeline you have run in the previous task.

   ![Review registered dataset in Azure Machine Learning studio.](media/devops-build-outputs-03.png "Registered Datasets in Azure Machine Learning studio")

4. Select the latest version of your dataset to review its properties. Notice the `build_number` tag that links the dataset version to the Azure DevOps build that generated it.

   ![Review registered dataset version properties, notice Build_Number tag.](medial/../media/devops-build-outputs-04.png "Registered dataset details in Azure Machine Learning studio")

5. Select **Models** to view a list of registered models that reference the dataset.

   ![Review list of registered models that reference dataset in Azure Machine Learning studio.](media/devops-build-outputs-05.png "Registered dataset model references in Azure Machine Learning studio")

## Exercise 5: Setup the Release Pipeline

Now that the train pipeline has succeeded, artifacts (model & `eval_info.json`) are available to set up the Release pipeline (sometimes called the deployment pipeline). Since we like the deployment to kick off directly after the train pipeline has succeeded, we use a release pipeline. Continuous integration is currently not possible within the pipeline section in Azure DevOps. Hence, instead of using a YAML file like we did for the IAC and Train pipeline, for the Release pipeline you will create these steps yourself. Furthermore, you will create variables required to perform the steps.

Duration: 20 minutes

### Task 1: Create an Empty Job

1. Return to Azure DevOps and navigate to **Pipelines, Releases** and select **New pipeline**.

   ![To create new Release Pipeline navigate to Pipelines, Releases and select New pipeline.](media/devops-release-pipeline-01.png "New Release Pipeline")

2. Select **Empty job**.

   ![Select empty job to start building the release pipeline.](media/devops-release-pipeline-02.png "Select a template: Empty job")

3. Provide Stage name: `Deploy & Test` and close the dialog.

   ![Provide stage name for the release stage.](media/devops-release-pipeline-03.png "Deploy & Test Stage")

### Task 2: Add Train Artifact

1. Select **Add an artifact**.

   ![Add a new artifact to the release pipeline.](media/devops-release-pipeline-04.png "Add an artifact")

2. Select Source type: `Build`, Source: `Train-Pipeline`. _Observe the note that shows that the train pipeline publishes the build artifact named devops-for-ai_. Finally, select **Add**.

   ![Provide information to add the build artifact.](media/attach-train-release.png "Add a build artifact")

### Task 3: Add Variables to Deploy & Test stage

1. Open **View stage tasks** link.

   ![Open view stage tasks link.](media/devops-release-pipeline-06.png "View Stage Tasks")

2. Open the **Variables** tab.

   ![Open variables tab.](media/devops-release-pipeline-07.png "Release Pipeline Variables")

3. Add four pipeline variables as name - value pairs and then select **Save** (use the default values in the **Save** dialog):

These variables are needed to deploy the model.

    a. Name: `aks_name` Value: `aks-cluster'

    b. Name: `aks_region` Value: should be the same region as the region of your Azure Machine Learning workspace (probably `westeurope`)

    c. Name: `service_name` Value: `compliance-service-` add here the number of your account name you used to log in on to this page (provided in the teams chat). When you click in the top right icon on this webpage you can view the account name.

    d. Name: `description` Value: `"Compliance Classifier Web Service"` (Note the double quotes around description value).

**Note**: - Keep the scope for the variables to `Deploy & Test` stage.

    - Make sure the service_name includes your the number of your account. The number of your account name you used to log in on to this page (provided in the teams chat). When you click in the top right icon of DevOps you can view the account name.

<!-- ![Add four pipeline variables as name value pairs and save.](media/devops-release-pipeline-08.png "Add Pipeline Variables") -->

### Task 4: Setup Agent Pool for Deploy & Test stage

1. Open the **Tasks** tab.

   ![Open view stage tasks link.](media/devops-release-pipeline-09.png "Pipeline Tasks")

2. Select **Agent job** and change **Agent pool** to `Azure Pipelines` and change **Agent Specification** to `ubuntu-16.04`.

   ![Change Agent pool to be Hosted Ubuntu 1604.](media/devops-release-pipeline-10.png "Agent Job Setup")

### Task 5: Add Use Python Version task

1. Select **Add a task to Agent job** (the **+** button), search for `Use Python Version`, and select **Add**.

   ![Add Use Python Version task to Agent job.](media/devops-release-pipeline-11.png "Add Use Python Version Task")

2. Provide **Display name:** `Use Python 3.6` and **Version spec:** `3.6`.

   ![Provide Display name and Version spec for the Use Python version task.](media/devops-release-pipeline-12.png "Use Python Version Task Dialog")

### Task 6: Add Install Requirements task

1. Select **Add a task to Agent job** (the **+** button), search for `Bash`, and select **Add**.

   ![Add Bash task to Agent job.](media/devops-release-pipeline-13.png "Add Bash Task")

2. Provide **Display name:** `Install Requirements` and select **object browser ...** to provide **Script Path**.

   ![Provide Display name for the Bash task.](media/devops-release-pipeline-14.png "Bash Task Dialog")

3. Navigate to **Linked artifacts/\_Train-Pipeline (Build)/devops-for-ai/environment_setup** and select **install_requirements.sh**.

   ![Provide Script Path to the Install Requirements bash file.](media/install-requirements.png "Select Path Dialog")

4. Expand **Advanced** and select **object browser ...** to provide **Working Directory**.

   ![Expand advanced section to provide Working Directory.](media/bash-advanced-workdir.png "Bash Task - Advanced Section")

5. Navigate to **Linked artifacts/\_Train-Pipeline (Build)/devops-for-ai** and select **environment_setup**.

   ![Provide path to the Working Directory.](media/environment-dir.png "Select Path Dialog")

### Task 7: Add Deploy & Test Webservice task

1. Select **Add a task to Agent job** (the **+** button), search for `Azure CLI`, and select **Add**.

   ![Add Azure CLI task to Agent job.](media/devops-release-pipeline-18.png "Azure CLI Task")

2. Provide the following information for the Azure CLI task:

   a. **Task version**: `1.*`

   b. **Display name**: `Deploy and Test Webservice`

   c. **Azure subscription**: `quick-starts-sc-aml`

   > **Note**: This is the service connection we created in Exercise 1 / Task 4.

   d. **Script Location**: `Inline script`

   e. **Inline Script**: `python aml_service/deploy.py --service_name $(service_name) --aks_name $(aks_name) --aks_region $(aks_region) --description $(description)`

   f. Expand **Advanced** and provide **Working Directory:** `$(System.DefaultWorkingDirectory)/_Train-Pipeline/devops-for-ai`.

   ![Setup the Azure CLI task using the information above.](media/sc-devops-release-pipeline.png "Azure CLI Task Dialog")

Please review the code in `aml_service/deploy.py`. This step will read `eval_info.json` and if the evaluation step decided to deploy the new trained model, it will deploy the new model to production in an **Azure Kubernetes Service (AKS)** cluster.

### Task 8: Define Deployment Trigger

1. Navigate to **Pipeline** tab, and select **Pre-deployment conditions** for the `Deploy & Test` stage.

2. Select **After release**.

   ![Setup Pre-deployment conditions for the Deploy & Test stage.](media/devops-release-pipeline-21.png "Pre-deployment Conditions Dialog")

3. Close the dialog.

### Task 9: Enable Continuous Deployment Trigger

1. Select **Continuous deployment trigger** for `_Train-Pipeline` artifact.

2. Enable: **Creates a release every time a new train artifact is available**.

   ![Enable Continuous Deployment Trigger for the Release pipeline.](media/devops-release-pipeline-22.png "Continuous Deployment Trigger Dialog")

3. Close the dialog

### Task 10: Save the Release Pipeline

1. Provide name: `Release-Pipeline`.

2. Select: **Save** (use the default values in the **Save** dialog).

   ![Provide name for the release pipeline and select save.](media/Release-pipeline.png "save")

## Exercise 6: Run the Release Pipeline

In this exercise you will execute the release pipeline and use the artifact from the previous train pipeline to deploy a model. Normally the release pipeline would be executed when the train pipeline has finished. Therefore, we would have to restart the train pipeline. Due to time constraints we will not do this, but instead trigger the release manually.

### Task 1: Start Release pipeline

1. Navigate to **Pipelines, Releases** and select **create release**

   ![Create release.](media/create-release.png "create release")

2. Select **Create**. Note that you can also activate a single automatic trigger here, instead of automatically triggering every time a new build artifact is available from the train pipeline (like we have done now)

   ![Create release.](media/create.png "Create")

### Task 2: Monitor Release Pipeline

1. Navigate to **Pipelines, Releases**. Observe that the Release pipeline is automatically triggered upon successful completion of the Train pipeline. Click on the button shown in the figure to view pipeline logs.

   ![Navigate to Pipelines, Releases and Select as shown in the figure to view pipeline logs.](media/devops-test-pipelines-05.png "Pipelines - Releases")

2. The release pipeline will run for about 5 minutes. Proceed to the next task when the release pipeline successfully completes.

## Exercise 7: Query the model via Postman

In this exercise, you will provide the model with data points and receive a prediction back. Now that we have a deployed model, an API call can be made.

Duration: 5 minutes

### Task 1: Get URI and Primary key of model

1. Open the Azure Machine Learning Platform [link](https://ml.azure.com/). Make sure you are in the Azure Machine Learning workspace that you created for this tutorial.

2. Go to the models tab and click on your latest model.

3. Find the model that you just created and then click on Endpoints

   ![View model in Azure Machine Learning studio.](media/sc_aml_model.png "Azure Machine Learning studio - Workspace, Deployments")

4. Click on the Endpoint attached to the model and click on Consume

   ![View endpoint in Azure Machine Learning studio.](media/sc_endpoint_model.png "Azure Machine Learning studio - endpoints")

5. Find the URI and the Primary key. You will need both in the next exercise so keep this window open.

   ![View endpoint and key in studio.](media/sc_key_URI.png "Azure Machine Learning studio - key, endpoint")

### Task 2: Query the deployed model via Postman

1. Open Postman (program that you installed prior to this tutorial. If you haven't installed Postman yet, please download and follow the instruction from the following [link](https://www.postman.com/downloads/))

2. Click on the + sign (top left corner)

<!-- ![Click on plus sign](media/plus-sign.png 'New request') -->

3. Select "POST" as the request type

4. Paste the scoring URI you copied from the previous task

5. Go to the authorization tab, choose the Bearer Token as type and paste the primary key from the previous task as token.

   ![Fill in postman settings](media/sc_post_request.png "Post request")

6. Go to the Body tab, select raw data and select JSON as its format.

7. Insert the following data: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 2, 5, 6, 4, 3, 1, 34]]

   ![Fill in postman settings](media/sc_send_request.png "Post request")

8. Press send and examine the output

9. If the models predicts a 0, the car complies to the government regulations. If 1 is predicted, the model predicts that the car not complies to the regulations and probably be an old car with a relatively high carbon footprint.

## Wrap-up

Congratulations on completing this tutorial.
If you want to continue practicing with ML Ops, we have created additional exercises. Find out about automation and continous integration of ML Ops with the following exercise [link](#Additional-Exercise-Test-train-and-Release-Pipelines).
If you no longer intend to use your Azure Portal subscriptions with your created resources, please do not forget to delete your resources. Instructions on how to delete your resources are provided in the following exercise [link](#Deletion-Exercise-Delete-resource-in-the-Azure-Portal)

To recap, in this tutorial you learned about:

1. Creating a new project in Azure DevOps.

2. Creating a IAC Pipeline to setup resources.

3. Creating a Train Pipeline to support model training.

4. Creating a Release Pipeline to support model deployment.

## Deletion Exercise: Delete resource in the Azure Portal

Don't forget to delete your resource on the Azure Portal if you don't have a free subscription. With a free subscription everything will automatically be deleted after 30 days, so you still have some time to practice after this tutorial.

If you don't have a free subscription and you would like to delete all your resources, do the following;

1. Open the [Azure Portal](https://portal.azure.com/)

2. Go to Resource Groups and find you Resource Group.

3. Click on "Delete Resource group" in the top menu bar.

## Take-Home Exercise: Test train and Release Pipelines

Now that we have set up the release pipeline, it can be tested by executing the train pipeline and checking whether the release is automatically triggered to deploy the model. You will execute the train pipeline, by changing some parameters of the algorithm in the training step. This change, will trigger the pipeline to be automatically executed. Once it is done, an artifact is available and the release is triggered. Note: the model will only be deployed if the trained model has a higher accuracy score than the current deployed model.

Duration: 30 minutes

### Task 1: Make Edits to Source Code

1. Navigate to: **Repos -> Files -> scripts -> `train.py`**.

2. **Edit** `train.py`.

3. Change the learning rate **(learning_rate)** of the model from **0.1** to **0.001**.

4. Change the number of estimators **(n_estimators)** from **10** to **30**.

5. Select **Commit**.

   ![Make edits to train.py by changing the learning rate. Select Commit after editing.](media/screenshot-lr-.png "Edit Train.py")

6. Provide comment: `Improving model performance: changed learning rate.` and select **Commit**.

   ![Provide commit comment for train.py.](media/devops-test-pipelines-02.png "Commit - Comment")

### Task 2: Monitor Train Pipeline

1. Navigate to **Pipelines, Pipelines**. Observe that the train pipeline is triggered because of the source code change.

   ![Navigate to Pipelines, Builds.](media/devops-test-pipelines-03.png "Pipelines - pipelines")

2. Select the pipeline run and monitor the pipeline steps. The pipeline will run for 16-18 minutes. After the model has been trained, check whether the release pipeline is triggered and the new model is deployed.

   ![Monitor train Pipeline. It will take around 15 minutes to complete.](media/devops-test-pipelines-04.png "Train Pipeline Steps")

### Additional resources and more information

To learn more about MLOps with the Azure Machine Learning service, visit the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-model-management-and-deployment)

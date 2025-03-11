import asyncio
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

# Import the workflow from your worker.py file (assuming it's in the same directory)
from worker import FetchSessionsWorkflow

async def main():
    # Connect to the Temporal server
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)

    # Run the workflow and get the result
    try:
        result = await client.execute_workflow(
            FetchSessionsWorkflow.run,  # Pass the workflow run function
            id="fetch-sessions-workflow-id",  # Use 'id=' keyword argument for workflow ID
            task_queue="fetch-sessions-queue",
        )

        print("Workflow result:")
        print(result) # Print the list of sessions

    except Exception as e:
        print(f"Error running workflow: {e}")

if __name__ == "__main__":
    asyncio.run(main())
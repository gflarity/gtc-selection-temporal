from temporalio.client import Client
from temporalio.worker import Worker
from workflows import FetchSessionsWorkflow
from activities import fetch_sessions, process_sessions, Session
import asyncio


async def run_worker():
    # Connect to the Temporal server (default: localhost:7233)
    client = await Client.connect("localhost:7233")

    # Create a worker that runs the workflow and activity
    worker = Worker(
        client,
        task_queue="fetch-sessions-queue",
        workflows=[FetchSessionsWorkflow], # Workflow class
        activities=[fetch_sessions, process_sessions], # Activity function
    )

    # Start the worker and run indefinitely
    print("Starting worker...")
    try:
        await worker.run()
    except asyncio.CancelledError:
        print("Worker shut down.")
    except Exception as e:
        print(f"Worker failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_worker())
from temporalio import workflow
from datetime import timedelta

with workflow.unsafe.imports_passed_through(): # Mark activities import as pass-through
    from activities import fetch_sessions, filter_sessions, process_sessions, Session, ProcessSessionsInput

@workflow.defn
class FetchSessionsWorkflow:
    @workflow.run
    async def run(self) -> list[Session]:
        min_sessions = []
        offset = None

        while True:

            print("fetching sessions")
            # Workflow ONLY calls the activity            
            new_sessions = await workflow.execute_activity(
                fetch_sessions, # Call the activity
                offset,
                start_to_close_timeout=timedelta(seconds=30),
            )
            if not new_sessions:
                print("no new sessions")
                break

            if offset is None:
                offset = len(new_sessions)
            else:            
                offset += len(new_sessions)

            print('offset in workflow', offset)

            # save some time by filtering out sessions that are not relevant
            filtered_sessions = await workflow.execute_activity(
                filter_sessions,
                new_sessions,
                start_to_close_timeout=timedelta(seconds=300),
            )

            print("filtered sessions ", len(new_sessions) - len(filtered_sessions))

            # Process the new sessions            
            min_sessions = await workflow.execute_activity(
                process_sessions,
                ProcessSessionsInput(min_sessions, filtered_sessions),
                start_to_close_timeout=timedelta(seconds=600),
            )

            print("processed sessions ", offset) 

        return min_sessions
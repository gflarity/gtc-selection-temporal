from temporalio import workflow
from datetime import timedelta
import os
from pydantic import BaseModel

with workflow.unsafe.imports_passed_through(): # Mark activities import as pass-through
    from activities import fetch_sessions, FetchSessionsInput, Session
    from activities import filter_sessions, FilterSessionsInput
    from activities import process_sessions, ProcessSessionsInput, ProcessSessionsInput
    from activities import filter_non_english_sessions, FilterNonEnglishSessionsInput


class TopTenGTCSessionsWorkflowInput(BaseModel):
    api_key: str
    

@workflow.defn
class TopTenGTCSessionsWorkflow:
    @workflow.run
    async def run(self, input: TopTenGTCSessionsWorkflowInput) -> list[Session]:
        api_key = input.api_key
        min_sessions = []
        offset = None

        while True:

            print("fetching sessions")
            # Workflow ONLY calls the activity            
            new_sessions = await workflow.execute_activity(
                fetch_sessions, # Call the activity
                FetchSessionsInput(from_offset=offset), # Pass the input
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

            # filter out non english sessions
            english_sessions = await workflow.execute_activity(
                filter_non_english_sessions,
                FilterNonEnglishSessionsInput(sessions=new_sessions),
                start_to_close_timeout=timedelta(seconds=5),
            )

            # save some time by filtering out sessions that are not relevant
            filtered_sessions = await workflow.execute_activity(
                filter_sessions,
                FilterSessionsInput(sessions=english_sessions, api_key=api_key),
                start_to_close_timeout=timedelta(seconds=300),
            )

            print("filtered sessions ", len(new_sessions) - len(filtered_sessions))

            # Process the new sessions            
            min_sessions = await workflow.execute_activity(
                process_sessions,
                ProcessSessionsInput(min_sessions=min_sessions, new_sessions=filtered_sessions, api_key=api_key),
                start_to_close_timeout=timedelta(seconds=600),
            )

            print("processed sessions ", offset) 

        return min_sessions
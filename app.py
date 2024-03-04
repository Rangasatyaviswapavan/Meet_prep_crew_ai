from textwrap import dedent
from crewai import Agent, Task, Crew
from tools.ExaSearchTool import ExaSearchTool
from tools.SerperTool import search
import streamlit as st

class MeetingPreparation():
    def __init__(self):
        self.participants = ""

    def get_participants(self):
        return self.participants

    def set_participants(self, participants):
        self.participants = participants

    def research_task(self, agent, context):
        return Task(
            description=dedent(f"""\
                Conduct comprehensive research on each of the individuals and companies
                involved in the upcoming meeting. Gather information on recent
                news, achievements, professional background, and any relevant
                business activities.

                Participants: {self.get_participants()}
                Meeting Context: {context}"""),
            expected_output=dedent("""\
                A detailed report summarizing key findings about each participant
                and company, highlighting information that could be relevant for the meeting."""),
            async_execution=True,
            agent=agent
        )

    def industry_analysis_task(self, agent, context):
        return Task(
            description=dedent(f"""\
                Analyze the current industry trends, challenges, and opportunities
                relevant to the meeting's context. Consider market reports, recent
                developments, and expert opinions to provide a comprehensive
                overview of the industry landscape.

                Participants: {self.get_participants()}
                Meeting Context: {context}"""),
            expected_output=dedent("""\
                An insightful analysis that identifies major trends, potential
                challenges, and strategic opportunities."""),
            async_execution=True,
            agent=agent
        )

    def meeting_strategy_task(self, agent, context, objective):
        return Task(
            description=dedent(f"""\
                Develop strategic talking points, questions, and discussion angles
                for the meeting based on the research and industry analysis conducted

                Meeting Context: {context}
                Meeting Objective: {objective}"""),
            expected_output=dedent("""\
                Complete report with a list of key talking points, strategic questions
                to ask to help achieve the meetings objective during the meeting."""),
            agent=agent
        )

    def summary_and_briefing_task(self, agent, context, objective):
        return Task(
            description=dedent(f"""\
                Compile all the research findings, industry analysis, and strategic
                talking points into a concise, comprehensive briefing document for
                the meeting.
                Ensure the briefing is easy to digest and equips the meeting
                participants with all necessary information and strategies.

                Meeting Context: {context}
                Meeting Objective: {objective}"""),
            expected_output=dedent("""\
                A well-structured briefing document that includes sections for
                participant bios, industry overview, talking points, and
                strategic recommendations."""),
            agent=agent
        )

    def create_agent(self, role, goal, tool, backstory):
        return Agent(
            role=role,
            goal=goal,
            tools=[tool],
            backstory=backstory,
            verbose=True
        )

meeting_prep = MeetingPreparation()

st.title('Meeting Prep')

participants = st.text_input("What are the emails for the participants (other than you) in the meeting?\n")
meeting_prep.set_participants(participants)

context = st.text_input("What is the context of the meeting?\n")
objective = st.text_input("What is your objective for this meeting?\n")

research_agent = meeting_prep.create_agent(
    role='Research Specialist',
    goal='Conduct thorough research on people and companies involved in the meeting',
    tool=search,
    backstory=dedent("""\
        As a Research Specialist, your mission is to uncover detailed information
        about the individuals and entities participating in the meeting. Your insights
        will lay the groundwork for strategic meeting preparation.""")
)

industry_analysis_agent = meeting_prep.create_agent(
    role='Industry Analyst',
    goal='Analyze the current industry trends, challenges, and opportunities',
    tool=search,
    backstory=dedent("""\
        As an Industry Analyst, your analysis will identify key trends,
        challenges facing the industry, and potential opportunities that
        could be leveraged during the meeting for strategic advantage.""")
)

meeting_strategy_agent = meeting_prep.create_agent(
    role='Meeting Strategy Advisor',
    goal='Develop talking points, questions, and strategic angles for the meeting',
    tool=search,
    backstory=dedent("""\
        As a Strategy Advisor, your expertise will guide the development of
        talking points, insightful questions, and strategic angles
        to ensure the meeting's objectives are achieved.""")
)

summary_and_briefing_agent = meeting_prep.create_agent(
    role='Briefing Coordinator',
    goal='Compile all gathered information into a concise, informative briefing document',
    tool=search,
    backstory=dedent("""\
        As the Briefing Coordinator, your role is to consolidate the research,
        analysis, and strategic insights.""")
)

research_task = meeting_prep.research_task(research_agent, context)
industry_analysis_task = meeting_prep.industry_analysis_task(industry_analysis_agent, context)
meeting_strategy_task = meeting_prep.meeting_strategy_task(meeting_strategy_agent, context, objective)
summary_and_briefing_task = meeting_prep.summary_and_briefing_task(summary_and_briefing_agent, context, objective)

crew = Crew(
    agents=[
        research_agent,
        industry_analysis_agent,
        meeting_strategy_agent,
        summary_and_briefing_agent
    ],
    tasks=[
        research_task,
        industry_analysis_task,
        meeting_strategy_task,
        summary_and_briefing_task
    ]
)

if st.button('Start Generation'):
    result = crew.kickoff()

    st.markdown(result)
    st.download_button(
        label="Download",
        data=result,
        file_name="meeting_prep.md",
        mime="text/plain"
    )

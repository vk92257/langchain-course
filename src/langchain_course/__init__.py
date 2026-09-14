import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

def main() -> None:
    print("Hello from langchain-course!")
    information = """

Arvind Kejriwal (Hindi pronunciation: [əɾʋɪn̪d̪ keːd͡ʒɾiːʋɑːl]; born 16 August 1968) is an Indian politician, activist and former bureaucrat, who served as the 7th Chief Minister of Delhi. He was the chief minister from 2013 to 2014 and from 2015 to 2024. He is also the national convener of the Aam Aadmi Party (AAP) since 2012. He represented the New Delhi constituency in the Delhi Legislative Assembly from 2015 to 2025, and previously from 2013 to 2014.

In 2006, Kejriwal was awarded the Ramon Magsaysay Award for his involvement in the Parivartan movement using right to information legislation in a campaign against government corruption. The same year, after resigning from government service, he founded the Public Cause Research Foundation to campaign for transparent governance. Before entering politics, Kejriwal had worked in the Indian Revenue Service. Prior to that, he was a mechanical engineer from IIT Kharagpur.

In 2012, he launched the AAP. In 2013, he assumed office as the Chief Minister of Delhi and resigned 49 days later over his inability to mobilise support for his proposed anti-corruption legislation. In the 2015 Delhi Legislative Assembly elections, the AAP registered an unprecedented majority. In subsequent 2020 elections, AAP re-emerged victorious and retained power in Delhi, following which, Kejriwal was sworn in as the Chief Minister of Delhi for the third time in a row. Outside Delhi, his party registered another major victory in 2022 Punjab Legislative Assembly election.

He was arrested on 21 March 2024 by the Enforcement Directorate on allegations of a liquor scam against the Aam Aadmi Party led Delhi Government.[1][2] He became the first ever sitting chief minister in India to be arrested.[3] His other party leaders, Satyendra Jain, Sanjay Singh and Manish Sisodia have also spent months to years in jail without bail, trial or conviction.[4] The opposition alliance called the arrest weeks before the 2024 Indian general election, a case of fabrication and "match-fixing" by the BJP. Amnesty International said that financial and terrorism laws have been weaponised to go after political opponents.[5] On 10 May, the Supreme Court ordered Kejriwal's release on interim bail until 1 June 2024, on account of campaigning for the election.[6][7] Kejriwal surrendered at Tihar Jail after the expiry of his bail period on 2 June 2024. On 13 September 2024, he was granted bail by Supreme Court with certain conditions, the case still continues.[8] On 17 September 2024, he resigned as Delhi Chief Minister saying he will only become CM again if he receives a public mandate.[9]

His party suffered a heavy defeat in the 2025 Delhi Legislative Assembly election, with he himself losing his seat to Parvesh Verma by a margin of over 4,000 votes from the New Delhi Assembly constituency along with many other notable AAP members. On 27 February 2026, he was granted a clean chit in the excise policy case, along with 22 others, including Manish Sisodia.[10]

    """

    summary_template = """
    give the information {information} about a person I want you to create:
    1. A Short Summary.
    2. two interesting facts about them.

    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-3.1-flash-lite",
    #     temperature=0.3,
    # )

    llm = ChatOllama(
            model="qwen3:4b",
            temperature=0.3,
        )


    chain = summary_prompt_template | llm
    result = chain.invoke({"information": information})

    print(result.content)


if __name__ == "__main__":
    main()

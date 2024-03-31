import argparse
import base64
import logging
import json
import os
import requests
import tqdm
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence, Tuple

from openai import OpenAI

from src import utils

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

client = OpenAI()

system_prompt = """
system_prompt
"""

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def save_document_to_json(
    document: Dict[str, Sequence], filename: str = "document_state.json"
):
    with open(filename, "w") as file:
        json.dump(document, file, indent=4)


def load_document_from_json(filename: str):
    try:
        with open(filename, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print("File not found. Starting with a new document.")
        return {"sections": [], "executive_summary": []}


def generate_text(
    prompt: str,
    logger: logging.Logger,
    json_mode: bool = False,
    api_engine: str = "gpt-4-0125-preview",
) -> str:
    if json_mode:
        completion = client.chat.completions.create(
            model=api_engine,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
    else:
        completion = client.chat.completions.create(
            model=api_engine,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
    return completion.choices[0].message.content


def generate_figure(
    prompt: str, logger: logging.Logger, fig_n: int, image_dir: str
) -> Dict[str, str]:
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        size="1024x1024",
        # quality="hd", # hd
        response_format="url",
    )

    # save the image
    generated_image_name = f"figure_{fig_n}.png"
    generated_image_filepath = os.path.join(image_dir, generated_image_name)
    generated_image_url = response.data[0].url  # extract image URL from response
    generated_image = requests.get(generated_image_url).content  # download the image

    with open(generated_image_filepath, "wb") as image_file:
        image_file.write(generated_image)  # write the image to the file

    # AI feedback on figure
    description = generate_text_from_vlm(
        "Describe this image concisely.",
        image_path=generated_image_filepath,
        engine="gpt-4-vision-preview",
    )
    caption = generate_text_from_vlm(
        f"Generate LaTeX code to insert {generated_image_name}, and give the appropriate caption describing the image. Respond only in LaTeX.",
        image_path=generated_image_filepath,
        engine="gpt-4-vision-preview",
    )
    return {
        "image_filepath": generated_image_filepath,
        "description": description,
        "caption": caption,
    }


def generate_text_from_vlm(
    prompt: str,
    logger: logging.Logger,
    image_path: str,
    engine: str = "gpt-4-vision-preview",
) -> str:
    base64_image = encode_image(image_path)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": engine,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ],
    }
    completion = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )

    return completion.json()["choices"][0]["message"]["content"]


def update_executive_summary(update: str, logger: logging.Logger, api_engine: str):
    summary = generate_text(
        f"Summarize this update for the executive summary highlighting the key story/points of the added paper content, in a few sentences or less, in a non-joking tone:\n{update}",
        engine=api_engine,
    )
    document["executive_summary"].append(summary)


def main_loop(
    document: Dict[str, Sequence],
    args: argparse.Namespace,
):
    # create a timestamped subdirectory for logging
    local_dir = Path(__file__).parent
    timestamp = datetime.now().strftime("%H_%M_%S")
    image_dir = local_dir / args.runs_dir / timestamp / args.image_dir
    doc_state_dir = local_dir / args.runs_dir / timestamp / args.doc_state_dir
    log_dir = local_dir / args.runs_dir / timestamp / args.log_dir

    # create the directories
    os.mkdir(image_dir)
    os.mkdir(doc_state_dir)
    os.mkdir(log_dir)

    logger = utils.create_logger(name="FOMO", log_dir=log_dir)
    logger.info("Running FOMO-writing")

    figure_count = 0
    doc_save_count = 0
    section_count = 0
    for _ in tqdm.tqdm(range(args.max_actions)):
        action_prompt = (
            "Executive Summary:\n"
            + "\n".join(document["executive_summary"])
            + "\n\nWhat is the most interesting action to take next, based on what you think would be most worthwhile doing to expand on the paper, following the current paper state hinted at in the executive summary? Consider proposing sections in the first few steps, and then filling in empty sections and generating figures at the same time, and once no part is empty, editing the existing figures and sections? If there are already some sections, please consider generating figures!"
            + ' Respond in json dict format only with the key "action" containing one of the following five options as a string:'
            + " [propose_section, write_section, edit_section, generate_figure, edit_figure]"
        )
        action_response = generate_text(
            action_prompt, logger, api_engine=args.engine, json_mode=True
        )
        action = json.loads(action_response)["action"]
        print(action)
        match action:
            case "propose_section":
                if section_count >= 7:
                    continue
                section_title_response = generate_text(
                    "Executive Summary:\n"
                    + "\n".join(document["executive_summary"])
                    + "\n\nPropose a new section title for the paper with a boring, standard name. If not yet mentioned in the executive summary, propose 'Abstract', and next time 'Introduction'. Respond in json format. It should be as short as possible, with a maximum of 5 words. The key 'title' should contain the answer",
                    logger,
                    api_engine=args.engine,
                    json_mode=True,
                )
                section_title = json.loads(section_title_response)["title"]
                title_already_exists = False
                for section in document["sections"]:
                    if section_title == section["title"]:
                        title_already_exists = True
                        break
                if title_already_exists:
                    continue
                document["sections"].append({"title": section_title, "content": ""})
                document["executive_summary"].append(
                    f"Added section heading to be filled in later: {section_title}"
                )
                section_count += 1
            case "write_section" if document["sections"]:
                # Decide on which section to choose to write, out of empty ones
                empty_sections = [
                    document["sections"][i]["title"]
                    for i in range(len(document["sections"]))
                    if document["sections"][i]["content"] == ""
                ]
                if len(empty_sections) == 0:
                    continue
                # Create a prompt out of the list of empty_sections, and have the output of generate_text propose the section to write next
                proposal_title_response = generate_text(
                    "Executive Summary:\n"
                    + "\n".join(document["executive_summary"])
                    + "\n\nYou'll be given a list of currently empty section titles."
                    + " Tell me in json dict format only which title section would be interesting to fill in next, following inspiration on the current state of the overall paper from the executive summary of the whole paper in the current state. The key 'chosen_title' should contain the answer in the form of a list index, starting from 0."
                    + f"\n\nList: {', '.join(empty_sections)}",
                    logger,
                    api_engine=args.engine,
                    json_mode=True,
                )
                proposal_title_index_in_empty_sections = json.loads(
                    proposal_title_response
                )["chosen_title"]

                if proposal_title_index_in_empty_sections >= len(empty_sections):
                    # Skip if invalid title chosen
                    continue
                title_index = None
                chosen_title_name = empty_sections[
                    proposal_title_index_in_empty_sections
                ]
                for section_index in range(len(document["sections"])):
                    if (
                        document["sections"][section_index]["title"]
                        == chosen_title_name
                    ):
                        title_index = section_index

                # Generate the section text
                content = generate_text(
                    "Executive Summary:\n"
                    + "\n".join(document["executive_summary"])
                    + f"\n\nWrite content for the section titled '{chosen_title_name}' following inspiration on the current state of the overall paper from the executive summary of the whole paper in the current state. Respond with only the LaTeX code for it.",
                    logger,
                    api_engine=args.engine,
                )
                document["sections"][title_index]["content"] = content
                update_executive_summary(
                    f"Wrote content for section: {document['sections'][title_index]['title']}.\nContent: {content}",
                    logger,
                    args.engine,
                )
            case "edit_section" if document["sections"]:
                # Decide on which section to choose to edit, out of filled ones
                filled_sections = [
                    document["sections"][i]["title"]
                    for i in range(len(document["sections"]))
                    if document["sections"][i]["content"] != ""
                    and "Figure " not in document["sections"][i]["title"]
                ]
                if len(filled_sections) == 0:
                    continue
                # Create a prompt out of the list of empty_sections, and have the output of generate_text propose the section to write next
                proposal_title_response = generate_text(
                    "Executive Summary:\n"
                    + "\n".join(document["executive_summary"])
                    + "\n\nYou'll be given a list of currently filled section titles."
                    + " Tell me in json dict format only which title section should be edited next, following inspiration on the current state of the overall paper from the executive summary of the whole paper in the current state. The key 'chosen_title' should contain the answer in the form of a list index, starting from 0."
                    + f"\n\nList: {', '.join(filled_sections)}",
                    logger,
                    api_engine=args.engine,
                    json_mode=True,
                )
                proposal_title_index_in_filled_sections = json.loads(
                    proposal_title_response
                )["chosen_title"]

                if proposal_title_index_in_filled_sections >= len(filled_sections):
                    # Skip if invalid title chosen
                    continue
                title_index = None
                chosen_title_name = filled_sections[
                    proposal_title_index_in_filled_sections
                ]
                for section_index in range(len(document["sections"])):
                    if (
                        document["sections"][section_index]["title"]
                        == filled_sections[proposal_title_index_in_filled_sections]
                    ):
                        title_index = section_index

                edited_content = generate_text(
                    "Executive Summary:\n"
                    + "\n".join(document["executive_summary"])
                    + f"\n\nRewrite this section following inspiration on the current state of the overall paper from the executive summary of the whole paper in the current state: {document['sections'][section_index]['content']}.\n\nRespond with only the LaTeX code for the rewritten version of this section.",
                    logger,
                    api_engine=args.engine,
                )
                document["sections"][title_index]["content"] = edited_content
                update_executive_summary(
                    f"Edited section: {document['sections'][title_index]['title']}.\nNew Content: {edited_content}",
                    logger,
                    args.engine,
                )
            case "generate_figure":
                if figure_count >= 5:
                    continue
                figure_count += 1
                image_gen_prompt_response = generate_text(
                    "Executive Summary:\n"
                    + "\n".join(document["executive_summary"])
                    + "\n\nYou're now tasked to create a text prompt as input to an image generation model."
                    + " Tell me in json dict format only which prompt should be used to get an image for an interesting figure for the current paper state, following inspiration from the executive summary of the whole paper in the current state. The key 'image_prompt' should contain the answer.",
                    logger,
                    api_engine=args.engine,
                    json_mode=True,
                )
                image_gen_prompt = json.loads(image_gen_prompt_response)["image_prompt"]
                output_figure_data = generate_figure(
                    image_gen_prompt, logger, figure_count, image_dir
                )
                document["sections"].append(
                    {
                        "title": f"Figure {figure_count}",
                        "content": output_figure_data["caption"],
                        "image_filepath": output_figure_data["image_filepath"],
                        "description": output_figure_data["description"],
                    }
                )
                update_executive_summary(
                    f"Generated Figure {figure_count} containing: {output_figure_data['description']}\nCaption: {output_figure_data['caption']}",
                    logger,
                    args.engine,
                )
            case "edit_figure":
                proposal_figure_number_response = generate_text(
                    "Executive Summary:\n"
                    + "\n".join(document["executive_summary"])
                    + f"\n\nInspired by the executive summary, propose which figure number (as an integer) would be interesting to edit and revise."
                    + f" So far, we have figures up to Figure {figure_count}. Tell me in json dict format only. The key 'figure_number' should contain the integer answer.",
                    logger,
                    api_engine=args.engine,
                    json_mode=True,
                )
                proposal_figure_number = json.loads(proposal_figure_number_response)[
                    "figure_number"
                ]
                # Get index number of document section containing target figure
                title_index = None
                for section_index in range(len(document["sections"])):
                    if (
                        document["sections"][section_index]["title"]
                        == f"Figure {proposal_figure_number}"
                    ):
                        title_index = section_index
                if title_index is None:
                    # Skip if invalid title chosen
                    continue
                image_gen_prompt_response = generate_text(
                    "Executive Summary:\n"
                    + "\n".join(document["executive_summary"])
                    + f"\n\nYou're now tasked to create a text prompt as input to an image generation model, for replacing Figure {proposal_figure_number}."
                    + " Tell me in json dict format only which prompt should be used to get an image for an interesting figure for the current paper state, following inspiration from the executive summary of the whole paper in the current state. The key 'image_prompt' should contain the answer.",
                    logger,
                    api_engine=args.engine,
                    json_mode=True,
                )
                image_gen_prompt = json.loads(image_gen_prompt_response)["image_prompt"]
                output_figure_data = generate_figure(
                    image_gen_prompt, logger, proposal_figure_number, image_dir
                )
                document["sections"][title_index] = {
                    "title": f"Figure {proposal_figure_number}",
                    "content": output_figure_data["caption"],
                    "image_filepath": output_figure_data["image_filepath"],
                    "description": output_figure_data["description"],
                }
                update_executive_summary(
                    f"Edited Figure {proposal_figure_number} now containing: {output_figure_data['description']}\nNew Caption: {output_figure_data['caption']}",
                    logger,
                    args.engine,
                )
            case _:
                print("Unknown action.")
        if len(document["executive_summary"]) > args.max_summary_items:
            document["executive_summary"].pop(0)
        # Save contents of generated paper to file, and add to counter
        doc_save_filename = os.path.join(doc_state_dir, f"doc_{doc_save_count}.json")
        save_document_to_json(document, doc_save_filename)
        doc_save_count += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load-doc-state-path",
        type=str,
        default=None,
        help="Optional path to load the document state from",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=200,
        help="Maximum number of actions in the main loop",
    )
    parser.add_argument(
        "--max-summary-items",
        type=int,
        default=50,
        help="Maximum number of items in the executive summary",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Parent directory to save timestamped run outputs",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Parent directory to save runtime logs",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="images",
        help="Parent directory to save images",
    )
    parser.add_argument(
        "--doc-state-dir",
        type=str,
        default="doc_states",
        help="Parent directory to save document states",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="gpt-4-0125-preview",
        help="API model to use for text generation ('gpt-4-0125-preview', 'gpt-3.5-turbo-0125')",
    )

    config, _ = parser.parse_known_args()
    return config


if __name__ == "__main__":
    args = parse_args()

    if args.load_doc_state_path:
        document = load_document_from_json(args.load_doc_state_path)
    else:
        document = {"sections": [], "executive_summary": []}

    main_loop(document, args)

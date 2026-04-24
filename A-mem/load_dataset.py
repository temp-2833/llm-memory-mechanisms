import json
from typing import Dict, List, Optional, Union
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class QA:
    question: str
    answer: Optional[str]
    evidence: List[str]
    category: Optional[int] = None
    adversarial_answer: Optional[str] = None

    @property
    def final_answer(self) -> Optional[str]:
        """Get the appropriate answer based on category."""
        if self.category == 5:
            return self.adversarial_answer
        return self.answer


@dataclass
class Turn:
    speaker: str
    dia_id: str
    text: str


@dataclass
class Session:
    session_id: int
    date_time: str
    turns: List[Turn]


@dataclass
class Conversation:
    speaker_a: str
    speaker_b: str
    sessions: Dict[int, Session]


@dataclass
class EventSummary:
    events: Dict[str, Dict[str, List[str]]]  # session -> speaker -> events


@dataclass
class Observation:
    observations: Dict[str, Dict[str, List[List[str]]]]  # session -> speaker -> [observation, evidence]


@dataclass
class LoCoMoSample:
    sample_id: str
    qa: List[QA]
    conversation: Conversation
    event_summary: EventSummary
    observation: Observation
    session_summary: Dict[str, str]


def parse_session(session_data: List[dict], session_id: int, date_time: str) -> Session:
    turns = []
    for turn in session_data:
        # For turns with images, combine caption and text
        text = turn.get("text", "")
        if "img_url" in turn and "blip_caption" in turn:
            caption_text = f"[Image: {turn['blip_caption']}]"
            if text:
                text = f"{caption_text} {text}"
            else:
                text = caption_text
            
        turns.append(Turn(
            speaker=turn["speaker"],
            dia_id=turn["dia_id"],
            text=text
        ))
    return Session(session_id=session_id, date_time=date_time, turns=turns)


def parse_conversation(conv_data: dict) -> Conversation:
    sessions = {}
    for key, value in conv_data.items():
        if key.startswith("session_") and isinstance(value, list):
            session_id = int(key.split("_")[1])
            date_time = conv_data.get(f"{key}_date_time")
            if date_time:
                session = parse_session(value, session_id, date_time)
                # Only add sessions that have turns after filtering
                if session.turns:
                    sessions[session_id] = session
    
    return Conversation(
        speaker_a=conv_data["speaker_a"],
        speaker_b=conv_data["speaker_b"],
        sessions=sessions
    )


def load_locomo_dataset(file_path: Union[str, Path]) -> List[LoCoMoSample]:
    if isinstance(file_path, str):
        file_path = Path(file_path)
        
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    total_qa = 0
    total_image_qa = 0
    qa_counts_per_sample = []
    
    for sample_idx, sample in enumerate(data):
        try:
            # Parse QA data
            qa_list = []
            sample_qa_count = 0
            sample_image_qa_count = 0
            
            for qa_idx, qa in enumerate(sample["qa"]):
                try:
                    has_image_evidence = False
                    for evidence_id in qa.get("evidence", []):
                        if ":" not in evidence_id:
                            continue
                        turn_id = evidence_id.split(":")[1]
                        for session in sample["conversation"].values():
                            if isinstance(session, list):
                                for turn in session:
                                    if turn.get("dia_id", "").endswith(turn_id):
                                        if "img_url" in turn or "blip_caption" in turn:
                                            has_image_evidence = True
                                            break
                    
                    if has_image_evidence:
                        sample_image_qa_count += 1
                        
                    qa_obj = QA(
                        question=qa["question"],
                        answer=qa.get("answer"),
                        evidence=qa.get("evidence", []),
                        category=qa.get("category"),
                        adversarial_answer=qa.get("adversarial_answer")
                    )
                    qa_list.append(qa_obj)
                    sample_qa_count += 1
                    
                except KeyError as e:
                    print(f"样本 {sample_idx} 发生错误, QA 对是 {qa_idx}:")
                    print(f"QA数据: {qa}")
                    raise e
                except Exception as e:
                    print(f"样本 {sample_idx} 发生错误, QA pair {qa_idx}:")
                    print(f"QA数据: {qa}")
                    raise e

            conversation = parse_conversation(sample["conversation"])

            event_summary = EventSummary(events=sample["event_summary"])

            observation = Observation(observations=sample["observation"])

            session_summary = sample.get("session_summary", {})

            sample_obj = LoCoMoSample(
                sample_id=str(sample_idx),
                qa=qa_list,
                conversation=conversation,
                event_summary=event_summary,
                observation=observation,
                session_summary=session_summary
            )
            samples.append(sample_obj)
            
            total_qa += sample_qa_count
            total_image_qa += sample_image_qa_count
            qa_counts_per_sample.append(sample_qa_count)
            
            # Print statistics for this sample
            print(f"\n样本{sample_idx}:")
            print(f"  总共的QA数量: {sample_qa_count}")
            print(f"  有图片的QA数量: {sample_image_qa_count}")
            
        except Exception as e:
            print(f"处理样本 {sample_idx} 时出现错误:")
            print(str(e))
            raise e
    
    # Print overall statistics
    print("\n整体数据:")
    print(f"QA数量: {total_qa}")
    print(f"有图片的QA数量: {total_image_qa}")
    print(f"每个样本平均QA数量: {total_qa / len(samples):.2f}")
    print(f"样本中的最小QA数量: {min(qa_counts_per_sample)}")
    print(f"样本中的最大QA数量: {max(qa_counts_per_sample)}")
    
    return samples


def get_dataset_statistics(samples: List[LoCoMoSample]) -> Dict:
    """
    Get basic statistics about the text-only dataset.
    
    Args:
        samples: List of LoCoMoSample objects
        
    Returns:
        Dictionary containing various statistics about the dataset
    """
    stats = {
        "num_samples": len(samples),
        "total_qa_pairs": sum(len(sample.qa) for sample in samples),
        "total_sessions": sum(len(sample.conversation.sessions) for sample in samples),
        "total_turns": sum(
            sum(len(session.turns) for session in sample.conversation.sessions.values())
            for sample in samples
        ),
        "qa_with_adversarial": sum(
            sum(1 for qa in sample.qa if qa.adversarial_answer is not None)
            for sample in samples
        )
    }
    return stats


if __name__ == "__main__":
    # Example usage
    dataset_path = Path(__file__).parent / "data" / "locomo10.json"
    try:
        print(f"从 {dataset_path} 加载数据集")
        samples = load_locomo_dataset(dataset_path)
        for sample_idx, sample in enumerate(samples):
            print(f"\n样本 {sample_idx}:")
            for _,turns in sample.conversation.sessions.items():
                for turn in turns.turns:
                    print(turn)
                    break   

    except Exception as e:
        print(f"加载数据集时出现错误: {e}")
        raise
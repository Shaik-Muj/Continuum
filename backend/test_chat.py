from sqlmodel import Session

from database import engine
from services.chat_service import chat


def main() -> None:
    with Session(engine) as session:
        response = chat(
            user_message="Where do I work?",
            user_id=1,
            session=session,
        )

        print("\nContinuum:\n")
        print(response)


if __name__ == "__main__":
    main()
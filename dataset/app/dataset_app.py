import typer
from dataset.app import split_command
from dataset.app import pre_process_command


FULL_TRAIN_SET = "full_training_set"
TRAIN_SET = "training_set"
VALIDATION_SET = "validation_set"
TEST_SET = "test_set"

SESSION_ID_KEY = "SessionId"
CLICKS_FILE_NAME = "yoochoose-clicks.dat"

app = typer.Typer()
# Register the split app with add_typer since it needs sub-commands
app.add_typer(split_command.app, name="split")
app.add_typer(pre_process_command.app, name="pre_process")


def get_dataset_app() -> typer.Typer:
    return app


# Python main function to make Typer-App available as CLI.
if __name__ == "__main__":
    app()

import typer
from dataset.app.commands import data_set_commands, index_command, split_commands


app = typer.Typer()
# Register the Typer sub-commands
app.add_typer(data_set_commands.app, name="pre_process")
app.add_typer(index_command.app, name="index")
app.add_typer(split_commands.app, name="split")


def get_data_set_app() -> typer.Typer:
    return app


# Python main function to make Typer-App available as CLI.
if __name__ == "__main__":
    app()

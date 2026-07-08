import typer

app = typer.Typer(help="pkm CLI")


@app.command()
def main() -> None:
    print("hello world")


if __name__ == "__main__":
    app()
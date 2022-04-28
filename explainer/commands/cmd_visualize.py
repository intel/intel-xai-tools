import click

@click.command("visualize", short_help="Processes an Explaination JSON definition and pass to the visualizer.")
@click.argument('input', default="-", type=click.File("w"), required=True)
def cli(input):
    click.echo(f"Input is {input}")


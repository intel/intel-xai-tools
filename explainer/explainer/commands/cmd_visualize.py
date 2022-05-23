import click

@click.command("visualize", short_help="Plots an explanation JSON definition.")
@click.argument('input', default="-", type=click.File("w"), required=True)
def cli(input):
    click.echo(f"Input is {input}")


import click
from flockrl_sim.cli.commands import generate


@click.group()
@click.version_option(version="0.1.0", prog_name="flockrl")
def cli():
    pass


cli.add_command(generate.generate)

if __name__ == "__main__":
    cli()

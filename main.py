import click


@click.group()
@click.argument('file_path', type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.pass_context
def cli(ctx, file_path):
    # TODO read yaml file and convert to parameters class
    pass


@cli.command()
@click.pass_context
def train(ctx):
    click.echo('Starting training...')


@cli.command()
@click.pass_context
def test(ctx):
    click.echo('Starting testing...')


@cli.command()
@click.pass_context
def predict(ctx):
    click.echo('Starting prediction...')


if __name__ == '__main__':
    cli(obj={})

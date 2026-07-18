from typer.testing import CliRunner

from pkm.new_agents.agent_000_dragapult.cli import app

runner = CliRunner()


def test_train_help_lists_method_and_mcts_options():
    res = runner.invoke(app, ["train", "--help"])
    assert res.exit_code == 0
    assert "--method" in res.output
    assert "--mcts-simulations" in res.output

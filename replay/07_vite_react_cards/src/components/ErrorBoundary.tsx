import { Component, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  /** Bump this (e.g. the step index) to clear the error when the input changes. */
  resetKey?: unknown;
}
interface State {
  error: Error | null;
}

// Keeps a single failing step from unmounting the whole app. When resetKey
// changes (user navigates to another step), the boundary clears itself.
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidUpdate(prev: Props) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) {
      this.setState({ error: null });
    }
  }

  render() {
    if (this.state.error) {
      return (
        <div className="board board-end">
          <div className="end-card">
            <h2>Couldn't render this step</h2>
            <p className="err-msg">{this.state.error.message}</p>
            <p>Use the controls to move to another step.</p>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

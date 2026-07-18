/*
  ErrorBoundary.tsx — a crash never white-screens HIDE. Catches a render error anywhere below, logs it,
  and shows a concrete recover panel (flight-log voice) instead of a blank page.
*/
import { Component, type ErrorInfo, type ReactNode } from "react";

export class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Local crash log (no egress). A crash-report file is the second-plan hardening item.
    console.error("HIDE render error:", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="crash" role="alert">
          <div className="crash__box glass">
            <div className="crash__title">HIDE hit a render error</div>
            <pre className="crash__detail">{this.state.error.message}</pre>
            <button className="crash__btn" onClick={() => location.reload()}>reload</button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

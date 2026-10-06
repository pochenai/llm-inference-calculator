const GITHUB_SUPPORT_KEY = 'llm-calc-github-support-unlocked';

export function readGithubSupportUnlocked(): boolean {
  try {
    return window.localStorage.getItem(GITHUB_SUPPORT_KEY) === '1';
  } catch {
    return false;
  }
}

export function persistGithubSupportUnlocked(): void {
  try {
    window.localStorage.setItem(GITHUB_SUPPORT_KEY, '1');
  } catch {
    // In-memory React state still unlocks both prompts for this visit.
  }
}

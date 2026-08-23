// Small form primitives shared by the input panels.

import type { ReactNode } from 'react';

export function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="card">
      <h3 className="card-title">{title}</h3>
      <div className="card-body">{children}</div>
    </section>
  );
}

export function Field({
  label,
  children,
  hint,
  error,
}: {
  label: string;
  children: ReactNode;
  hint?: string | undefined;
  error?: string | undefined;
}) {
  return (
    <div className="field">
      <span className="field-label">{label}</span>
      {children}
      {error ? <span className="field-error">{error}</span> : null}
      {!error && hint ? <span className="field-hint">{hint}</span> : null}
    </div>
  );
}

export function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step,
  placeholder,
  hint,
  error,
  disabled,
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  min?: number | undefined;
  max?: number | undefined;
  step?: number | undefined;
  placeholder?: string | undefined;
  hint?: string | undefined;
  error?: string | undefined;
  disabled?: boolean | undefined;
}) {
  return (
    <Field label={label} hint={hint} error={error}>
      <input
        className="input num"
        type="number"
        min={min}
        max={max}
        step={step}
        value={Number.isFinite(value) ? value : ''}
        placeholder={placeholder}
        disabled={disabled}
        onChange={(e) => onChange(e.target.valueAsNumber)}
      />
    </Field>
  );
}

export function SelectField({
  label,
  value,
  onChange,
  options,
  hint,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
  hint?: string | undefined;
}) {
  return (
    <Field label={label} hint={hint}>
      <select className="input" value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </Field>
  );
}

export function Toggle({
  label,
  desc,
  checked,
  onChange,
}: {
  label: string;
  desc?: string | undefined;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <button
      type="button"
      className={`toggle${checked ? ' on' : ''}`}
      aria-pressed={checked}
      onClick={() => onChange(!checked)}
    >
      <span className="toggle-track">
        <span className="toggle-thumb" />
      </span>
      <span className="toggle-text">
        {label}
        {desc ? <small>{desc}</small> : null}
      </span>
    </button>
  );
}

// Row of compact chip buttons (used for quantization).
export function ChipGroup<T extends string>({
  label,
  value,
  onChange,
  options,
  hint,
}: {
  label: string;
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: string; sub?: string | undefined }[];
  hint?: string | undefined;
}) {
  return (
    <Field label={label} hint={hint}>
      <div className="chip-row">
        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            className={`chip${o.value === value ? ' active' : ''}`}
            onClick={() => onChange(o.value)}
          >
            {o.label}
            {o.sub ? <small>{o.sub}</small> : null}
          </button>
        ))}
      </div>
    </Field>
  );
}

export function CollapseSection({
  title,
  summary,
  open,
  onToggle,
  children,
}: {
  title: string;
  summary?: string | undefined;
  open: boolean;
  onToggle: () => void;
  children: ReactNode;
}) {
  return (
    <section className="card">
      <button type="button" className="card-title collapse-head" onClick={onToggle}>
        <span className="collapse-caret">{open ? '▾' : '▸'}</span>
        {title}
        {summary ? <span className="collapse-summary">{summary}</span> : null}
      </button>
      {open ? <div className="card-body">{children}</div> : null}
    </section>
  );
}

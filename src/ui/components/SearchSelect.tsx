// Searchable single-select dropdown (used for model / GPU catalogs).
// Spaces in the query act as wildcards: "qwen 4b" matches "Qwen3.5 4B".

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useI18n } from '../lib/i18n';

export interface SearchOption {
  id: string;
  name: string;
  sub?: string; // secondary line with key specs
  tag?: string; // small badge, e.g. MoE / Dense
  category?: string; // for filtering via category chips
}

export interface SearchCategory {
  value: string;
  label: string;
}

interface Props {
  options: SearchOption[];
  value: string;
  onChange: (id: string) => void;
  placeholder?: string;
  // When provided, a row of filter chips is shown above the search input.
  // "All" is prepended automatically.
  categories?: SearchCategory[];
  // When provided, a small copy button appears next to the input field.
  copyText?: string;
}

const MAX_RENDER = 400;

// Compact clipboard button with brief ✅ feedback after a successful copy.
function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const { t } = useI18n();
  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(text).then(
      () => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1200);
      },
      () => {
        // ignore clipboard failures silently
      },
    );
  }, [text]);
  return (
    <button
      type="button"
      className={`copy-btn${copied ? ' copied' : ''}`}
      onClick={handleCopy}
      title={t('label.copy')}
      aria-label={t('label.copy')}
    >
      {copied ? '✅' : '📋'}
    </button>
  );
}

export function SearchSelect({ options, value, onChange, placeholder, categories, copyText }: Props) {
  const { t } = useI18n();
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);
  const [activeCat, setActiveCat] = useState<string | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);

  const current = options.find((o) => o.id === value);

  const filtered = useMemo(() => {
    let list = options;

    // Apply category filter first
    if (activeCat) {
      list = list.filter((o) => o.category === activeCat);
    }

    // Spaces act as wildcards: "qwen 4b" → /qwen.*4b/i → matches "Qwen3.5 4B"
    const q = query.trim();
    if (!q) return list;
    const escaped = q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const re = new RegExp(escaped.replace(/\s+/g, '.*'), 'i');
    return list.filter((o) => re.test(`${o.name} ${o.id}`));
  }, [options, query, activeCat]);

  useEffect(() => {
    function onDocPointerDown(e: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener('mousedown', onDocPointerDown);
    return () => document.removeEventListener('mousedown', onDocPointerDown);
  }, []);

  const hasCats = categories && categories.length > 0;

  return (
    <div className="search-select" ref={rootRef}>
      {hasCats && (
        <div className="size-chip-row">
          <button
            type="button"
            className={`size-chip${activeCat === null ? ' active' : ''}`}
            onClick={() => setActiveCat(null)}
          >
            {t('label.all')}
          </button>
          {categories.map((c) => (
            <button
              key={c.value}
              type="button"
              className={`size-chip${activeCat === c.value ? ' active' : ''}`}
              onClick={() => setActiveCat(c.value)}
            >
              {c.label}
            </button>
          ))}
        </div>
      )}
      <div className="search-select-row">
        <input
          className="input"
          value={open ? query : current?.name ?? ''}
          placeholder={placeholder ?? t('placeholder.search')}
          onFocus={() => {
            setOpen(true);
            setQuery('');
          }}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Escape') setOpen(false);
          }}
        />
        {copyText !== undefined && <CopyButton text={copyText} />}
      </div>
      {open && (
        <div className="search-list">
          {filtered.slice(0, MAX_RENDER).map((o) => (
            <button
              key={o.id}
              type="button"
              className={`search-item${o.id === value ? ' active' : ''}`}
              onClick={() => {
                onChange(o.id);
                setOpen(false);
              }}
            >
              <span className="search-name">
                {o.name}
                {o.tag ? <em className={`tag${o.tag === 'MoE' ? ' tag-moe' : ''}`}>{o.tag}</em> : null}
              </span>
              {o.sub ? <span className="search-sub">{o.sub}</span> : null}
            </button>
          ))}
          {filtered.length > MAX_RENDER && (
            <div className="search-empty">{t('note.more_items', { count: filtered.length - MAX_RENDER })}</div>
          )}
          {filtered.length === 0 && <div className="search-empty">{t('note.no_match')}</div>}
        </div>
      )}
    </div>
  );
}

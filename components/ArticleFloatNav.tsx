'use client'

export default function ArticleFloatNav() {
  return (
    <div className="pointer-events-none fixed right-3 bottom-4 z-40 flex flex-col gap-1">
      <button
        type="button"
        aria-label="回到顶端"
        onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
        className="pointer-events-auto h-7 w-7 border border-gray-300/40 bg-white/20 font-mono text-[10px] text-gray-500 opacity-25 transition-opacity hover:opacity-80"
      >
        顶
      </button>
      <button
        type="button"
        aria-label="回到底端"
        onClick={() =>
          window.scrollTo({ top: document.documentElement.scrollHeight, behavior: 'smooth' })
        }
        className="pointer-events-auto h-7 w-7 border border-gray-300/40 bg-white/20 font-mono text-[10px] text-gray-500 opacity-25 transition-opacity hover:opacity-80"
      >
        底
      </button>
    </div>
  )
}

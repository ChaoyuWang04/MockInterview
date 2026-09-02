import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

const eslintConfig = [
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    // 与 .gitignore 保持一致:凡是 git 不追踪的目录,lint 也不该扫。
    // 尤其 projects/ —— 那是第三方源码仓(pytorch/vllm 等),只是撰写开源解读时的
    // 阅读材料,不是本项目代码;不排掉它 npm run lint 会被 3 万条噪音淹没。
    ignores: [
      "node_modules/**",
      ".next/**",
      ".next-prod/**",
      ".leet/**",
      "out/**",
      "build/**",
      "coverage/**",
      "projects/**",
      "next-env.d.ts",
    ],
  },
  {
    rules: {
      // `_` 前缀 = 故意解构出来丢掉的,不是死代码。
      // 典型场景:components/Markdown.tsx 里 `code({ node: _node, ...props })` ——
      // 不把 node 摘出来,它会跟着 ...props 铺到 DOM 上,React 报未知属性警告。
      "@typescript-eslint/no-unused-vars": [
        "warn",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          caughtErrorsIgnorePattern: "^_",
          destructuredArrayIgnorePattern: "^_",
          ignoreRestSiblings: true,
        },
      ],
    },
  },
];

export default eslintConfig;

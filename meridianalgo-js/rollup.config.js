const typescript = require('rollup-plugin-typescript2');
const { nodeResolve } = require('@rollup/plugin-node-resolve');
const commonjs = require('@rollup/plugin-commonjs');
const json = require('@rollup/plugin-json');
const terser = require('@rollup/plugin-terser');

const pkg = require('./package.json');

const external = Object.keys(pkg.dependencies || {});

module.exports = [
  // CommonJS build
  {
    input: 'src/index.ts',
    output: {
      file: pkg.main,
      format: 'cjs',
      sourcemap: true,
      exports: 'named'
    },
    external,
    plugins: [
      nodeResolve(),
      commonjs(),
      json(),
      typescript({
        typescript: require('typescript'),
        tsconfig: './tsconfig.json',
        useTsconfigDeclarationDir: true
      })
    ]
  },
  // ES Module build
  {
    input: 'src/index.ts',
    output: {
      file: pkg.module,
      format: 'es',
      sourcemap: true
    },
    external,
    plugins: [
      nodeResolve(),
      commonjs(),
      json(),
      typescript({
        typescript: require('typescript'),
        tsconfig: './tsconfig.json',
        useTsconfigDeclarationDir: true
      })
    ]
  },
  // UMD build for browsers
  {
    input: 'src/index.ts',
    output: {
      file: 'dist/meridianalgo-js.umd.js',
      format: 'umd',
      name: 'MeridianAlgo',
      sourcemap: true,
      globals: {
        'ml-matrix': 'MLMatrix',
        'ml-regression': 'MLRegression',
        'simple-statistics': 'ss',
        'lodash': '_'
      }
    },
    external,
    plugins: [
      nodeResolve(),
      commonjs(),
      json(),
      typescript({
        typescript: require('typescript'),
        tsconfig: './tsconfig.json',
        useTsconfigDeclarationDir: true
      })
    ]
  },
  // Minified UMD build
  {
    input: 'src/index.ts',
    output: {
      file: 'dist/meridianalgo-js.umd.min.js',
      format: 'umd',
      name: 'MeridianAlgo',
      sourcemap: true,
      globals: {
        'ml-matrix': 'MLMatrix',
        'ml-regression': 'MLRegression',
        'simple-statistics': 'ss',
        'lodash': '_'
      }
    },
    external,
    plugins: [
      nodeResolve(),
      commonjs(),
      json(),
      typescript({
        typescript: require('typescript'),
        tsconfig: './tsconfig.json',
        useTsconfigDeclarationDir: true
      }),
      terser()
    ]
  }
];
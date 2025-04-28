- https://github.com/jfavre/ISC25.git

# Repo structure

├── components -> slidev scripts
├── node_modules -> see below (slidev deps)
├── public -> images
├── slides.md -> main slide
├── slidev-theme-cscs -> theme from https://github.com/eth-cscs/slidev-theme-cscs.git with minor changes
├── snippets -> code examples
└── src -> slides

# Setup

- Install `slidev` with:

```bash
npm init -y
npm install @slidev/cli
```

This will install a large dir `node_modules` with required slidev deps.

# Slides

- Launch the slides with:

```bash
npm install create-slidev@51.6.0
npm run dev
```

# References

- https://eth-cscs.github.io/cug25-uenv/
- https://eth-cscs.github.io/swe4py/slides/1
- https://sli.dev/guide/

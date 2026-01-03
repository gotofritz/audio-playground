# Coverage Badge

After the first successful push to `main`, you can add the coverage badge to your README:

```markdown
![Coverage](https://raw.githubusercontent.com/gotofritz/audio-playground/badges/coverage.svg)
```

## How it works

1. On every push to `main`, the workflow generates a coverage badge SVG
2. The badge is stored in the `badges` branch (auto-created)
3. You can embed the badge in README or other documentation
4. The badge updates automatically on each main branch push

## Badge colors

- **Green (≥80%)** - Excellent coverage
- **Orange (60-79%)** - Moderate coverage
- **Red (<60%)** - Poor coverage

## PR Coverage Comments

Pull requests automatically receive comments showing:
- Overall coverage percentage
- Coverage change compared to base branch
- Files with coverage changes
- Uncovered lines

# OMR Development Notes

## Project Overview
Building an Optical Mark Recognition (OMR) system for Typst-generated quizzes that:
- Works with variable number of questions (no fixed templates)
- Uses corner brackets for bubble region detection
- Generates test data for validation

## Key Design Decisions

### 1. Marker System
**Choice: Corner Brackets**
- Using L-shaped brackets at corners of bubble grid
- 12pt size, 2pt thickness in Typst
- Provides clear visual boundaries for OpenCV detection
- Better than QR/ArUco (simpler) or full borders (less intrusive)

### 2. Detection Strategy (Template-Free)
- No JSON templates needed - dynamic detection
- Find brackets → Extract region → Detect circles → Cluster to grid
- Multiple fallback strategies if brackets not found

## Technical References

### Typst Drawing
- Use `line()` instead of deprecated `path()`
- `place()` for absolute positioning
- `box()` to contain positioned elements
- Context blocks for dynamic content based on question count

### OpenCV Techniques (from research)
1. **From PyImageSearch tutorial**:
   - Contour detection with aspect ratio filtering (0.9-1.1)
   - Pixel counting in masks for fill detection
   - Four-point perspective transform

2. **From OMRChecker (Udayraj123)**:
   - Morphological operations for alignment
   - Auto grid detection from bubble positions
   - 30% fill threshold for marking

3. **From my old LaTeX code**:
   - Template matching for registration
   - Hough lines for rotation correction
   - Binary bubble reading

### Python OMR Pipeline
```python
1. detect_corner_brackets() - Find L-shapes
2. extract_bubble_region() - Perspective transform
3. detect_all_bubbles() - HoughCircles or contours
4. cluster_to_grid() - Group by rows/columns
5. analyze_fill() - Count pixels, threshold at 30%
```

## Test Scenarios to Generate
- [x] Basic: 5, 10, 15, 20 questions
- [ ] Edge cases: No answers, all answers
- [ ] Partial fills (various percentages)
- [ ] Rotation: ±5 degrees
- [ ] Noise: Gaussian, salt & pepper
- [ ] Scan artifacts: Blur, shadows

## File Structure
```
typst-quiz-omr/
├── quiz_template.typ      # Modified with brackets
├── omr/                   # Python OMR code
│   ├── detector.py       # Main detection
│   ├── markers.py        # Bracket detection
│   └── bubbles.py        # Bubble analysis
├── tests/
│   ├── generate_tests.typ # Test data generation
│   └── test_scans/        # Generated PDFs
└── omr_development_notes.md # This file
```

## Progress Update

### Completed
1. ✅ Added corner brackets to Typst template
2. ✅ Created test generator with 5 test cases (5, 10, 15, 20 questions, various patterns)
3. ✅ Set up Python project with uv
4. ✅ Built initial OMR detector with:
   - Corner bracket detection (needs tuning)
   - Fallback bubble region detection
   - Grid clustering algorithm
   - Fill ratio analysis
5. ✅ Created CLI tool for testing

### Current Challenges
- Corner bracket detection not working reliably at 300 DPI
- Brackets might be too thin (2pt) or too small (12pt) 
- Need to tune detection parameters for high-res scans
- Consider making brackets thicker/larger in Typst

### Next Steps
1. Improve bracket detection:
   - Make brackets thicker in Typst (try 4pt stroke, 20pt size)
   - Tune template matching parameters
   - Add morphological operations to enhance brackets
2. Test bubble detection separately
3. Add rotation correction
4. Generate test images with simulated scan artifacts

## Useful Commands
```bash
# Compile quiz
typst compile quiz.typ

# Watch for changes
typst watch quiz.typ

# Future: Run OMR
python omr/detect.py quiz.pdf
```

## Performance Targets
- 200+ sheets/minute (from OMRChecker benchmark)
- 90%+ accuracy on mobile scans
- 100% accuracy on clean scans
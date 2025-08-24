'use strict';

(function initializeSnakeGame() {
  const canvas = document.getElementById('game');
  const ctx = canvas.getContext('2d');

  const CELL_SIZE = 24;
  const GRID_COLS = 24;
  const GRID_ROWS = 24;
  const CANVAS_WIDTH = CELL_SIZE * GRID_COLS;
  const CANVAS_HEIGHT = CELL_SIZE * GRID_ROWS;

  canvas.width = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;

  const COLORS = {
    background: '#0d1326',
    snakeBody: '#67e8f9',
    snakeHead: '#22d3ee',
    food: '#fb7185',
    grid: 'rgba(255,255,255,0.04)'
  };

  const HIGHSCORE_KEY = 'snake_high_score';

  /** @typedef {{x: number, y: number}} Cell */

  /** State */
  /** @type {Cell[]} */
  let snakeSegments = [];
  /** @type {Cell} */
  let foodCell = { x: 0, y: 0 };
  /** @type {Cell} */
  let currentDirection = { x: 1, y: 0 };
  /** @type {Cell} */
  let queuedDirection = { x: 1, y: 0 };
  let isPaused = false;
  let isGameOver = false;
  let score = 0;
  let highScore = Number(localStorage.getItem(HIGHSCORE_KEY) || '0') || 0;

  let lastTimeMs = 0;
  let accumulatorMs = 0;
  let tickIntervalMs = 120;

  const scoreEl = document.getElementById('score');
  const highScoreEl = document.getElementById('highscore');
  const pauseBtn = document.getElementById('pause-btn');
  const restartBtn = document.getElementById('restart-btn');
  const overlay = document.getElementById('overlay');
  const overlayText = document.getElementById('overlay-text');

  ctx.imageSmoothingEnabled = false;

  function positionsEqual(a, b) {
    return a.x === b.x && a.y === b.y;
  }

  function areOpposite(a, b) {
    return a.x === -b.x && a.y === -b.y;
  }

  function setOverlay(message) {
    if (!message) {
      overlay.classList.add('hidden');
      overlay.setAttribute('aria-hidden', 'true');
      overlayText.textContent = '';
      return;
    }
    overlay.classList.remove('hidden');
    overlay.setAttribute('aria-hidden', 'false');
    overlayText.innerHTML = message;
  }

  function updateHud() {
    scoreEl.textContent = String(score);
    highScoreEl.textContent = String(highScore);
    pauseBtn.textContent = isPaused ? 'Resume' : 'Pause';
    pauseBtn.setAttribute('aria-pressed', isPaused ? 'true' : 'false');
  }

  function randomIntInclusive(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }

  function randomFreeCell() {
    while (true) {
      const candidate = { x: randomIntInclusive(0, GRID_COLS - 1), y: randomIntInclusive(0, GRID_ROWS - 1) };
      if (!snakeSegments.some(s => positionsEqual(s, candidate))) {
        return candidate;
      }
    }
  }

  function spawnFood() {
    foodCell = randomFreeCell();
  }

  function resetGame() {
    const startX = Math.floor(GRID_COLS / 2);
    const startY = Math.floor(GRID_ROWS / 2);
    snakeSegments = [
      { x: startX, y: startY },
      { x: startX - 1, y: startY },
      { x: startX - 2, y: startY }
    ];
    currentDirection = { x: 1, y: 0 };
    queuedDirection = { x: 1, y: 0 };
    isPaused = false;
    isGameOver = false;
    score = 0;
    tickIntervalMs = 120;
    spawnFood();
    setOverlay('');
    updateHud();
  }

  function handleKeydown(event) {
    const key = event.key;
    let handled = false;

    if (key === 'ArrowLeft' || key === 'a' || key === 'A') {
      queueDirection({ x: -1, y: 0 }); handled = true;
    } else if (key === 'ArrowRight' || key === 'd' || key === 'D') {
      queueDirection({ x: 1, y: 0 }); handled = true;
    } else if (key === 'ArrowUp' || key === 'w' || key === 'W') {
      queueDirection({ x: 0, y: -1 }); handled = true;
    } else if (key === 'ArrowDown' || key === 's' || key === 'S') {
      queueDirection({ x: 0, y: 1 }); handled = true;
    } else if (key === ' ' || key === 'Spacebar' || key === 'Space' || key === 'p' || key === 'P') {
      togglePause(); handled = true;
    } else if (key === 'r' || key === 'R') {
      resetGame(); handled = true;
    }

    if (handled) {
      event.preventDefault();
    }
  }

  function queueDirection(newDirection) {
    if (isGameOver) return;
    if (areOpposite(newDirection, currentDirection)) return;
    queuedDirection = newDirection;
  }

  function togglePause() {
    if (isGameOver) return;
    isPaused = !isPaused;
    setOverlay(isPaused ? '<strong>Paused</strong><br/>Press Space to resume' : '');
    updateHud();
  }

  function gameOver() {
    isGameOver = true;
    setOverlay('<strong>Game Over</strong><br/>Press R to restart');
  }

  function update() {
    currentDirection = queuedDirection;

    const head = snakeSegments[0];
    const nextHead = { x: head.x + currentDirection.x, y: head.y + currentDirection.y };

    if (nextHead.x < 0 || nextHead.y < 0 || nextHead.x >= GRID_COLS || nextHead.y >= GRID_ROWS) {
      gameOver();
      return;
    }

    if (snakeSegments.some((segment, index) => index !== 0 && positionsEqual(segment, nextHead))) {
      gameOver();
      return;
    }

    const ateFood = positionsEqual(nextHead, foodCell);

    snakeSegments.unshift(nextHead);
    if (ateFood) {
      score += 1;
      if (score > highScore) {
        highScore = score;
        localStorage.setItem(HIGHSCORE_KEY, String(highScore));
      }
      if (tickIntervalMs > 70) {
        tickIntervalMs -= 2;
      }
      spawnFood();
      updateHud();
    } else {
      snakeSegments.pop();
    }
  }

  function drawRoundedRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function render() {
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    ctx.fillStyle = COLORS.food;
    const fx = foodCell.x * CELL_SIZE;
    const fy = foodCell.y * CELL_SIZE;
    drawRoundedRect(fx + 4, fy + 4, CELL_SIZE - 8, CELL_SIZE - 8, 6);
    ctx.fill();

    for (let i = snakeSegments.length - 1; i >= 0; i--) {
      const segment = snakeSegments[i];
      const sx = segment.x * CELL_SIZE;
      const sy = segment.y * CELL_SIZE;
      const isHead = i === 0;
      ctx.fillStyle = isHead ? COLORS.snakeHead : COLORS.snakeBody;
      drawRoundedRect(sx + 2, sy + 2, CELL_SIZE - 4, CELL_SIZE - 4, isHead ? 8 : 6);
      ctx.fill();
    }
  }

  function step(timestamp) {
    if (!lastTimeMs) lastTimeMs = timestamp;
    const delta = timestamp - lastTimeMs;
    lastTimeMs = timestamp;

    if (!isPaused && !isGameOver) {
      accumulatorMs += delta;
      while (accumulatorMs >= tickIntervalMs) {
        accumulatorMs -= tickIntervalMs;
        update();
      }
    }

    render();
    requestAnimationFrame(step);
  }

  pauseBtn.addEventListener('click', () => {
    if (isGameOver) return;
    togglePause();
  });

  restartBtn.addEventListener('click', () => {
    resetGame();
  });

  document.addEventListener('keydown', handleKeydown, { passive: false });

  resetGame();
  updateHud();
  requestAnimationFrame(step);
})();
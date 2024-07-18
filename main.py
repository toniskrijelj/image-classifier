from torch import nn
import pygame
import grid
import image_classifier
import labels as ls

pygame.init()
screen = pygame.display.set_mode([1000, 800])

classifier = [(ls.digit_labels, 'digit_model', 'Digits'),
              (ls.sketch_labels1, 'sketch_model1', 'Sketches 1'),
              (ls.sketch_labels2, 'sketch_model2', 'Sketches 2'),
              (ls.sketch_labels3, 'sketch_model3', 'Sketches 3')]

labels = classifier[0][0]
font_size = 25 + 20 - len(labels)
my_font = pygame.font.SysFont('Comic Sans MS', font_size)
size = len(labels)
grid_size = 28
grid_len = 20
grid_start = 100
grid = grid.Grid(grid_size, grid_size, grid_start, grid_start, grid_len, grid_len)
clf = image_classifier.ImageClassifier('cpu', 1, 28, 28, size)
clf.load(classifier[0][1])
label_x = grid_start + grid_size * grid_len + 50
label_y_mid = grid_start + grid_len * grid_size // 2
label_size = font_size * 1.1
label_y_start = label_y_mid - len(labels) // 2 * label_size
label_y_end = label_y_mid + (len(labels) + 1) // 2 * label_size
label_step = (label_y_end - label_y_start) // len(labels)
mode = 0


def load_classifier():
    global label_y_start, label_y_end, label_step, label_size, my_font, font_size, size, clf, labels
    labels = classifier[mode][0]
    size = len(labels)
    font_size = 25 + 20 - len(labels)
    my_font = pygame.font.SysFont('Comic Sans MS', font_size)
    clf = image_classifier.ImageClassifier('cpu', 1, 28, 28, size)
    clf.load(classifier[mode][1])
    label_size = font_size * 1.1
    label_y_start = label_y_mid - len(labels) // 2 * label_size
    label_y_end = label_y_mid + (len(labels) + 1) // 2 * label_size
    label_step = (label_y_end - label_y_start) // len(labels)


def print_probabilities():
    tensor = grid.matrix.unsqueeze(0).unsqueeze(0).to('cpu')
    yhat = clf.calculate(tensor)
    yhat = nn.functional.softmax(yhat, dim=1)
    arr = []
    for i in range(size):
        arr.append((yhat[0][i].item(), i))
    arr.sort(reverse=True)
    for i in range(size):
        col = (100, 100, 100)
        if i == 0:
            col = (240, 240, 240)
        text_surface = my_font.render(labels[arr[i][1]] + " " + str(round(arr[i][0] * 100, 2)) + "%", False, col)
        screen.blit(text_surface, (label_x, label_y_start + label_step * i))


def print_info():
    font = pygame.font.SysFont('Comic Sans MS', 20)
    col = (10, 10, 10)
    text = font.render('Left Mouse: Paint | Right Mouse: Erase | C: Clear', False, col)
    screen.blit(text, (150, 680))
    text = font.render('Left / Right Key: Change classifier', False, col)
    screen.blit(text, (200, 710))


def print_heading():
    font = pygame.font.SysFont('Comic Sans MS', 50)
    col = (30, 30, 30)
    text = font.render(classifier[mode][2], False, col)
    screen.blit(text, (270, 20))


def paint():
    background_color = (240, 168, 92)
    screen.fill(background_color)
    grid.redraw_full()
    print_probabilities()
    print_info()
    print_heading()
    pygame.display.flip()


paint()

FPS = 500
clock = pygame.time.Clock()
running = True
while running:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                grid.clear()
                paint()
            if event.key == pygame.K_RIGHT:
                mode = (mode + 1) % len(classifier)
                load_classifier()
                paint()
            if event.key == pygame.K_LEFT:
                mode = (mode - 1 + len(classifier)) % len(classifier)
                load_classifier()
                paint()

        if event.type == pygame.MOUSEBUTTONUP:
            paint()

    grid.try_paint()
    clock.tick(FPS)

pygame.quit()

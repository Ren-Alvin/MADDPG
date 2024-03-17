class map:
    def __init__(self, n_width, n_height):
        self.n_width = n_width
        self.n_height = n_height
        self.typemap = [[0 for _ in range(n_width)] for _ in range(n_height)]

    def render(self):
        import pyglet
        from pyglet.window import key

        window = pyglet.window.Window(width=self.n_width * 40, height=self.n_height * 40)

        @window.event
        def on_key_press(symbol, modifiers):
            # 检测到按键事件后关闭窗口
            window.close()

        @window.event
        def on_draw():
            window.clear()

            # 绘制格子
            for y in range(self.n_height):
                for x in range(self.n_width):
                    cell_type = self.typemap[y][x]
                    color = (255, 255, 255) if cell_type == 0 else (0, 0, 0)  # 空格子是白色，障碍物是黑色
                    x1, y1 = x * 40, y * 40
                    x2, y2 = x1 + 40, y1 + 40
                    pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [x1, y1, x2, y1, x2, y2, x1, y2]),
                                         ('c3B', color * 4))

            # 绘制线条以区分格子
            for x in range(self.n_width + 1):
                pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2f', [x * 40, 0, x * 40, self.n_height * 40]),
                                     ('c3B', (0, 0, 0) * 2))
            for y in range(self.n_height + 1):
                pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2f', [0, y * 40, self.n_width * 40, y * 40]),
                                     ('c3B', (0, 0, 0) * 2))

        pyglet.app.run()


m=map(5,5)
m.typemap[1][1]=1
m.render()
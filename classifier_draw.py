# 带有分类器的绘图
from tensorflow import keras
from draw import *


class ClassifierDraw(DrawWindow):
    def __init__(self, model: keras.Model, label_list: np.ndarray|tuple[str]|list[str], img_process_func: Callable, top_number:int=10, debug_func_list:list|None=None):
        super().__init__()
        
        # 传入模型和标签列表以及图片预处理的函数
        if debug_func_list is None:
            debug_func_list = []
        self.debug_func_list = debug_func_list
        self.model = model
        self.label_list = label_list
        self.img_process_func = img_process_func
        
        # 增加统计输出
        self.stat_frame = tk.Frame(self.window, width=100, height=512, highlightthickness=1, highlightbackground='#909090')
        self.stat_frame.grid(row=0, column=2, padx=5, pady=5, sticky=tk.N)
        self.top_prob_title = tk.Label(self.stat_frame, text='top-{} prob:'.format(top_number), justify=tk.LEFT)
        self.top_prob_title.grid(row=0, column=0, padx=5, pady=5, sticky=tk.NW)
        self.top_probs = []
        for i in range(top_number):
            t_label = tk.Label(self.stat_frame, justify=tk.LEFT)
            t_label.grid(row=i+1, column=0, padx=5, pady=5, sticky=tk.NW)
            self.top_probs.append(t_label)
        
        # debug
        if len(debug_func_list) != 0:
            self.debug = tk.Button(self.stat_frame, command=self._debug, highlightthickness=0)
            self.debug.grid(row=top_number+3, column=0, padx=5, pady=5, sticky=tk.S)
    
    
    # 每次更新时获取预测结果
    def _draw_graph_refreshed(self):
        x = self.img_process_func(self.get_draw())
        y = self.model.predict(np.expand_dims(x, axis=0), verbose=0, batch_size=1)[0]
        index_list = np.argsort(y)[::-1]
        for i in range(len(self.top_probs)):
            self.top_probs[i]['text'] = '  {}: {:.2%}'.format(self.label_list[index_list[i]], y[index_list[i]])
            
    # 保存和加载
    def save(self, name:str|None=None):
        if name is None:
            name = self.title
        self.model.save("_{}/model.keras".format(name))
        np.save("_{}/label.npy".format(name), self.label_list)
        
    @staticmethod
    def load(img_process_func: Callable, name='Draw'):
        model = keras.models.load_model("_{}/model.keras".format(name))
        label_list = np.load("_{}/label.npy".format(name))
        return ClassifierDraw(model, label_list, img_process_func)
    
    
    def _debug(self):
        for func in self.debug_func_list:
            func(self)


def test_train_and_show():
    from model import get_model
    from data import DataKyrgyz
    
    # 数据读取，注意图像的文件夹默认名称为 dataset
    DATA = DataKyrgyz('dataset')
    # 获取数据
    x_train, y_train = DATA.get_train_data()
    
    # 带有 BN 的模型
    model_BN = get_model(32, dropout=False, BN=True)
    model_BN.compile(loss='categorical_crossentropy', optimizer='sgd')
    
    # 学习率策略
    def lr_scheduler(epoch):
        if epoch < 4: return 0.1
        return 0.4 / (epoch + 1)
    
    lr_callback = keras.callbacks.LearningRateScheduler(lr_scheduler)
    
    # 训练
    model_BN.fit(x_train, y_train, epochs=4, batch_size=32, callbacks=[lr_callback], verbose=0)
    
    # 得到交互式窗口
    window = ClassifierDraw(model_BN, DATA.get_label_list(), DATA.image_process)
    window.show()


def test_load_and_show():
    from data import DataKyrgyz
    
    window = ClassifierDraw.load(DataKyrgyz.image_process)
    window.show()

if __name__ == "__main__":
    test_load_and_show()
    
from dataset.process import HacREDDataset, SQuADDataset
import copy

class ExtractionEnv:
    def __init__(self, llm_func, data_path, dataset, lang, mode='train', data_split=0.8):
        self.data = None
        self.state = None
        self.llm_func = llm_func
        self.data_split = data_split
        self.mode = mode
        self.dataname = dataset
        self.index = 0
        self.lang = lang
        self.join = ''
        
        if dataset == 'HacRED':
            self.dataset = HacREDDataset(data_path=data_path, data_split=self.data_split)
        elif dataset == 'SQuAD2.0':
            self.dataset = SQuADDataset(data_path=data_path, data_split=self.data_split)
            
        self.train_data = self.dataset.train_datas
        self.test_data = self.dataset.test_datas
        if self.mode == 'train':
            self.dataset_len = len(self.train_data)
        else:
            self.dataset_len = len(self.test_data)

    def step(self, cond, action, choices, rl_indices):
        new_content = choices[action].strip()

        if not cond:
            new_join = new_content
        else:
            if self.lang == 'zh':
                new_join = cond + new_content
            elif self.lang == 'en':
                new_join = cond + ' ' + new_content
        score = self.reward_assign(rl_indices)

        new_choices = copy.deepcopy(choices)
        del new_choices[action]
        if new_choices:
            done = False
        else:
            done = True

        new_cond = f'{new_join}'
        self.join = new_join
        return (new_cond, new_choices), score, done
    
    def reward_assign(self, rl_indices):
        if self.dataname == 'HacRED' or self.dataname == 'SQuAD2.0':
            n = len(rl_indices)
            if self.target[:n] == rl_indices:
                return 1
            else:
                return 0

    def return_cond(self):
        return self.join

    def reset(self):
        if self.dataname == 'HacRED' or self.dataname == 'SQuAD2.0':
            if self.mode == 'train':
                index = self.index % self.dataset_len
                self.index += 1
                self.data = self.train_data[index]
            else:
                index = self.index
                self.index += 1
                self.data = self.test_data[index]

            self.sentence_list = self.data[0]
            self.paragraph = self.data[2]
            self.target = self.data[3]
            self.mapping = self.data[4]
            choices = self.sentence_list
            
            return ('', choices), 0, False
    
    def choose(self, cond, choices, prompt_mode):
        if len(choices) == 1:
            return 0
        if prompt_mode == 'prompt_no_icl' and self.lang=='zh':
            prompt = f'''现在你是一个文字专家，需要完成连句成段任务。现在我将给你已经拼接好的部分【Incomplete Paragraph】，和一个句子候选列表【choices】。
请你在【choices】中选出一个最合适的句子连接在【Incomplete Paragraph】后面，返回这个句子在【choices】列表中的位置。比如，你选择的是【choices】的第一句话，就直接返回1，你选择的是【choices】的第二句话，就直接返回2，以此类推。
即使你觉得没有合适的句子，也必须选择一个最合适的。
注意，仅仅返回一个数字代表位置，不得有除数字外的其他输出，且这个数字必须严格在1和{len(choices)}之间，千万注意输出不能大于{len(choices)}。
'''
        elif prompt_mode == 'prompt_icl' and self.lang=='zh':
            prompt = f'''现在你是一个文字专家，需要完成连句成段任务。现在我将给你已经拼接好的部分【Incomplete Paragraph】，和一个句子候选列表【choices】。
请你在【choices】中选出一个最合适的句子连接在【Incomplete Paragraph】后面，返回这个句子在【choices】列表中的位置。比如，你选择的是【choices】的第一句话，就直接返回1，你选择的是【choices】的第二句话，就直接返回2，以此类推。
即使你觉得没有合适的句子，也必须选择一个最合适的。下面我将给你三个例子：
Example1:
【Incomplete Paragraph】:苏宝连教授1983年毕业于辽宁大学化学系，1986年在中科院成都有机所获得硕士学位，1992年获得法国巴黎大学博士学位，1995年在比利时那摩尔大学从事博士后研究工作；之后，曾在美国催化公司工作，现在他在比利时那摩尔大学工作，担任那摩尔大学化学系副主任，是国际著名的从事无机材料、分子筛与催化研究的学者；同时，他还是多种国际杂志的审稿人。
【choices】: ['曾多年受邀作为诺贝尔化学奖推荐人。', '2002年9月被破格晋升为无机材料化学教授，2004年再一次被破格晋升为终身教授，2006年创建了纳米化学中心。']
Output:2
Example2:
【Incomplete Paragraph】:LCD Soundsystem美国纽约的另类舞曲乐队，在音乐创作、人声与器乐录制以及合成制作上由唱片公司DFA Records的联合创始人之一，James Murphys一个人参与，而到现场演出时，则由一支完整的乐队形式出现。
【choices】: ['主要作品有《Losing My Edge》、《Yeah》等。', '乐队发行过三张专辑：首张同名专辑《LCD Soundsystem》（2005）、《Sound of Silver》 （2007）和 《This Is Happening》 （2010），均获得了评论界的赞扬，最后一张还跟他们带来了商业上的成功，占据了美国公告牌200强（Billboard 200） 以及英国专辑销量榜（UK Albums Chart）的前十。', '乐队曾三次获得格莱美奖的提名。']
Output:1
Example3:
【Incomplete Paragraph】:《好想大声说爱你》（君が好きだと叫びたい）是日本乐团BAAD于1993年12月1日推出的一张的单曲，也是该乐团的第3张单曲。
【choices】: ['由山田恭二作词，多々纳好夫谱曲。', '该曲同时作为日本动画片《灌篮高手》第1季的片头曲，为BAAD最受欢迎的一首歌。', '动画片《灌篮高手》的次回预告都会使用这首歌的无声版当背景音乐，到了第2季也继续使用。']
Output:1
注意，仅仅返回一个数字代表位置，不得有除数字外的其他输出，且这个数字必须严格在1和{len(choices)}之间。现在，请你完成这个任务。
'''
        elif prompt_mode == 'prompt_no_icl' and self.lang=='en':
            prompt = f'''Now you are a text expert and need to complete the task of forming coherent paragraphs. I will provide you with a partially assembled paragraph 【Incomplete Paragraph】 and a list of sentence candidates 【choices】.
Please select the most suitable sentence from 【choices】 to connect after the 【Incomplete Paragraph】, and return the position of this sentence in the 【choices】 list. For example, if you choose the first sentence in 【choices】, simply return 1; if you choose the second sentence, return 2, and so on.
Even if you think none of the sentences fit well, you must choose the one that fits best.
Note that you should return only a single number representing the position, and this number must strictly be between 1 and {len(choices)}. Remember, there must be no output other than this number.
'''
        elif prompt_mode == 'prompt_icl' and self.lang=='en':
            prompt = f'''Now you are a text expert and need to complete the task of forming coherent paragraphs. I will provide you with a partially assembled paragraph 【Incomplete Paragraph】 and a list of sentence candidates 【choices】.
Please select the most suitable sentence from 【choices】 to connect after the 【Incomplete Paragraph】, and return the position of this sentence in the 【choices】 list. For example, if you choose the first sentence in 【choices】, simply return 1; if you choose the second sentence, return 2, and so on.
Even if you think none of the sentences fit well, you must choose the one that fits best. Here are three examples:
Example1:
【Incomplete Paragraph】: Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress.
【choices】: ['Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time.', 'Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child.']
Output:2
Example2:
【Incomplete Paragraph】:Following the disbandment of Destiny's Child in June 2005, she released her second solo album, B'Day (2006), which contained hits \"Déjà Vu\", \"Irreplaceable\", and \"Beautiful Liar\".
【choices】: ['Beyoncé took a hiatus from music in 2010 and took over management of her career; her fourth album 4 (2011) was subsequently mellower in tone, exploring 1970s funk, 1980s pop, and 1990s soul.', 'Sasha Fierce (2008), which saw the birth of her alter-ego Sasha Fierce and earned a record-setting six Grammy Awards in 2010, including Song of the Year for \"Single Ladies (Put a Ring on It)\".', 'Beyoncé also ventured into acting, with a Golden Globe-nominated performance in Dreamgirls (2006), and starring roles in The Pink Panther (2006) and Obsessed (2009).']
Output:3
Example3:
【Incomplete Paragraph】:
【choices】: ['Beyoncé's name is a tribute to her mother's maiden name.', 'Beyoncé's younger sister Solange is also a singer and a former member of Destiny's Child.', 'Beyoncé Giselle Knowles was born in Houston, Texas, to Celestine Ann \"Tina\" Knowles (née Beyincé), a hairdresser and salon owner, and Mathew Knowles, a Xerox sales manager.', 'Mathew is African-American, while Tina is of Louisiana Creole descent (with African, Native American, French, Cajun, and distant Irish and Spanish ancestry).', 'Through her mother, Beyoncé is a descendant of Acadian leader Joseph Broussard.']
Output:3
Note that you should return only a single number representing the position, and this number must strictly be between 1 and {len(choices)}. Remember at all times, no output other than the number is allowed.
'''
        
        prompt += '【Incomplete Paragraph】: ' + cond + '\n【choices】: ' + str(choices) + '\nOutput:'
        position = self.llm_func(prompt)
        position = position.split('\n')[0].strip()
        position = position.strip("'")
        return int(position) - 1
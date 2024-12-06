import pandas as pd
from jiwer import wer, cer
import sacrebleu
import ast


class MetricEvaluator:
    def __init__(self, true_file, prediction_file):
        """
        Инициализация класса.
        :param true_file: путь к файлу с правильными ответами (CSV или TSV)
        :param prediction_file: путь к файлу с предсказаниями модели (CSV или TSV)
        """
        self.true_csv = self.read_file(true_file)
        self.pred_csv = self.read_file(prediction_file)
        self.validate_data()
    
    def read_file(self, file_path):
        """
        Чтение файла с определением разделителя (CSV или TSV).
        :param file_path: путь к файлу
        :return: DataFrame
        """
        try:
            return pd.read_csv(file_path, sep=",")  # Читаем как CSV
        except pd.errors.ParserError:
            return pd.read_csv(file_path, sep="\t")  # Если ошибка, читаем как TSV

    def validate_data(self):
        """
        Проверка совместимости данных.
        """
        if self.true_csv.columns.tolist() != self.pred_csv.columns.tolist():
            raise ValueError("Столбцы true_csv и pred_csv не совпадают.")

        if len(self.true_csv) != len(self.pred_csv):
            raise ValueError("Количество строк true_csv и pred_csv не совпадает.")

    def calculate_metrics_by_id(self):
        """
        Расчет метрик для каждого ID.
        :return: DataFrame с метриками для каждого ID.
        """
        # Создаем датафейм для результатов 
        result_df = self.true_csv.drop(columns='answear_bbox') # Удаляем максимально избыточную информацию. Можно еще удалить "images_names"

        # Создаем хранилища для метрик 
        wer_error_list = []
        cer_error_list = []
        bleu_score_list = []

        for row in range(len(self.true_csv)):
            # Преобразуем строку в список. Пример: "['Ответ 1', 'Ответ 2']" -> ['Ответ 1', 'Ответ 2']
            Y_true = ast.literal_eval(self.true_csv['answers'][row])
            y_pred = ast.literal_eval(self.pred_csv['answers'][row])

            # Проверка на кол-во ответов 
            if len(Y_true) != len(y_pred):
                pass # Что-то надо придумать или не надо...
                
            # Вычисляем метрику WER
            wer_error = wer(Y_true, y_pred)
            wer_error_list.append(wer_error)

            # Вычисляем метрику CER
            cer_error = cer(Y_true, y_pred)
            cer_error_list.append(cer_error)

            # Вычисляем метрику BLEU
            bleu_score = sacrebleu.corpus_bleu(Y_true, [y_pred]).score
            bleu_score_list.append(bleu_score)

        # Дополняем наш результирующий датафрейм метриками		
        metrics = {
          'pred_answers': self.pred_csv['answers'],
          'wer_error': wer_error_list,
          'cer_error': cer_error_list,
          'bleu_score': bleu_score_list
        }
        metrics = pd.DataFrame(metrics)
        result_df = pd.concat([result_df, metrics], axis=1)

        return result_df

    def calculate_metrics_by_doc_type(self, df):
        """
        Расчет метрик для каждого типа документа.
        :param df: DataFrame из метода calculate_metrics_by_id
        :return: DataFrame с метриками для каждого типа документа.
        """
        # Создаем список из всех уникальных типов документов
        doc_types = list(df['doc_class'].value_counts().index)

        # Создаем хранилища для метрик 
        wer_error_list = []
        cer_error_list = []
        bleu_score_list = []

        # Фильтруем и аппендим хранилища 
        for doc_type in doc_types:
            wer_error_list.append(df[df['doc_class'] == doc_type]['wer_error'].mean())
            cer_error_list.append(df[df['doc_class'] == doc_type]['cer_error'].mean())
            bleu_score_list.append(df[df['doc_class'] == doc_type]['bleu_score'].mean())

        # Создаем DataFrame с метриками для каждого типа документа
        doc_type_metrics = {
            'doc_class': doc_types,
            'wer_error': wer_error_list,
            'cer_error': cer_error_list,
            'bleu_score': bleu_score_list
        }

        return pd.DataFrame(doc_type_metrics)

    def group_by_doc_question(self, df): 
        """
        Группировка по тип документа + тип вопроса.
        :param df: pandas.DataFrame - исходный датафрейм
        :return: pandas.DataFrame - сгруппированный датафрейм с метриками
        """
        grouped = df.groupby(['doc_class', 'question_type'])['wer_error', 'cer_error', 'bleu_score'].mean().reset_index()
        
        return grouped

    def calculate_metrics_general(self):
        """
        Расчет метрик по общему корпусу данных.
        :return: Словарь с метриками WER, CER и BLEU.
        """
        true_answers = []
        pred_answers = []
        
        for row in range(len(self.true_csv)):
            true_answers.extend(ast.literal_eval(self.true_csv['answers'][row]))
            pred_answers.extend(ast.literal_eval(self.pred_csv['answers'][row]))
        
        wer_error = wer(true_answers, pred_answers)
        cer_error = cer(true_answers, pred_answers)
        bleu_score = sacrebleu.corpus_bleu(true_answers, [pred_answers]).score

        return {
            'wer_error': wer_error,
            'cer_error': cer_error,
            'bleu_score': bleu_score
        }